use crate::bridge::{self, AudioFormats, BridgeInfo, BridgeReceiver, BridgeSender, Capabilities};
use crate::feedback;
use base64::{Engine, engine::general_purpose::STANDARD};
use canary_agent_kernel::{ChatMessage, ModelFunctionCall, TokenUsage, TurnItemKind};
use canary_agent_runtime::{
    AgentError, AudioChunk, AudioEncoding, AudioFormat, ModelClient, ModelDescriptor, ModelInput,
    ModelInteraction, ModelResponseStatus, ModelSession, ModelSessionCommand, ModelSessionConfig,
    ModelSessionEvent, ModelSessionFuture, ModelSessionReceiver, ModelSessionSender, Result,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc};
use tokio::sync::{Mutex, mpsc};

#[derive(Clone)]
pub struct BridgeConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub voice: String,
    pub python: String,
    pub script: PathBuf,
}

impl BridgeConfig {
    pub fn validate(&mut self) -> Result<()> {
        if self.api_key.is_empty() || self.model.is_empty() {
            return Err(AgentError::Model(
                "realtime API key and model are required".into(),
            ));
        }
        let mut url = url::Url::parse(&self.endpoint)
            .map_err(|_| AgentError::Model("invalid realtime endpoint".into()))?;
        let scheme = match url.scheme() {
            "https" | "wss" => "wss",
            "http" | "ws" => "ws",
            _ => {
                return Err(AgentError::Model(
                    "realtime endpoint requires http(s) or ws(s)".into(),
                ));
            }
        };
        url.set_scheme(scheme)
            .map_err(|_| AgentError::Model("invalid realtime URL scheme".into()))?;
        if !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(AgentError::Model("realtime endpoint must not contain credentials, query, or fragment; set REAL_TIME_MODEL separately".into()));
        }
        self.endpoint = url.to_string();
        Ok(())
    }

    pub fn input_format(&self) -> AudioFormat {
        audio_format(16000)
    }
}

fn audio_format(sample_rate: u32) -> AudioFormat {
    AudioFormat {
        encoding: AudioEncoding::PcmS16Le,
        sample_rate,
        channels: 1,
    }
}

pub struct BridgeClient {
    config: BridgeConfig,
    worker: PathBuf,
    playback: Option<mpsc::UnboundedSender<feedback::Event>>,
}

impl BridgeClient {
    pub fn new(config: BridgeConfig) -> Self {
        Self {
            worker: config.script.clone(),
            config,
            playback: None,
        }
    }

    pub fn with_playback(mut self) -> (Self, feedback::Events) {
        let (sender, receiver) = mpsc::unbounded_channel();
        self.playback = Some(sender);
        (self, receiver)
    }
}

struct Sender {
    bridge: Arc<Mutex<BridgeSender>>,
    capabilities: Option<Capabilities>,
}
struct Receiver {
    bridge: BridgeReceiver,
    output_rate: u32,
    writer: feedback::Writer,
    playback: Option<mpsc::UnboundedSender<feedback::Event>>,
    connection_id: String,
    playback_feedback: bool,
}

impl ModelClient for BridgeClient {
    fn model_descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            fqn: self.config.model.clone(),
            settings: json!({"voice":self.config.voice,"transport":"python-jsonl"}),
        }
    }
    fn interaction(&self) -> ModelInteraction {
        ModelInteraction::Bidirectional
    }
    fn connect(&self, config: ModelSessionConfig) -> ModelSessionFuture<'_, ModelSession> {
        Box::pin(async move {
            let (sender, receiver) = bridge::spawn(&self.config.python, &self.worker)?;
            let writer = Arc::new(Mutex::new(sender));
            let mut sender = Sender {
                bridge: writer.clone(),
                capabilities: None,
            };
            let mut receiver = Receiver {
                bridge: receiver,
                output_rate: 0,
                writer,
                playback: self.playback.clone(),
                connection_id: config.connection_id.clone(),
                playback_feedback: false,
            };
            let instructions = config
                .request
                .messages
                .iter()
                .filter_map(|m| match m {
                    ChatMessage::System { content } => Some(content.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            let history: Vec<_> = config
                .request
                .messages
                .iter()
                .filter(|m| !matches!(m, ChatMessage::System { .. }))
                .collect();
            sender.write(json!({"type":"connect","endpoint":self.config.endpoint,"api_key":self.config.api_key,
                "model":self.config.model,"voice":self.config.voice,"instructions":instructions,
                "tools":config.request.functions,"history":history})).await?;
            match tokio::time::timeout(std::time::Duration::from_secs(30), receiver.read())
                .await
                .map_err(|_| AgentError::Model("realtime bridge setup timed out".into()))??
            {
                Some(Event::Ready {
                    provider,
                    audio,
                    capabilities,
                }) => {
                    if let (Some(provider), Some(audio), Some(capabilities)) =
                        (provider, audio, capabilities)
                    {
                        let info = BridgeInfo {
                            provider,
                            audio,
                            capabilities,
                        };
                        info.validate()?;
                        receiver.output_rate = info.audio.output_rate;
                        receiver.playback_feedback = info.capabilities.playback_feedback;
                        receiver.emit(feedback::Event::Ready {
                            connection_id: config.connection_id,
                            provider: info.provider,
                            tool_calls: info.capabilities.tool_calls,
                            text_input: info.capabilities.text_input,
                        });
                        sender.capabilities = Some(info.capabilities);
                    } else {
                        return Err(AgentError::Model(
                            "realtime bridge did not provide complete capability metadata".into(),
                        ));
                    }
                    Ok(ModelSession {
                        sender: Box::new(sender),
                        receiver: Box::new(receiver),
                    })
                }
                Some(Event::Error { message }) => Err(AgentError::Model(message)),
                _ => Err(AgentError::Model(
                    "realtime bridge closed or sent an unexpected setup event".into(),
                )),
            }
        })
    }
}

impl Sender {
    async fn write(&mut self, value: Value) -> Result<()> {
        self.bridge.lock().await.write(&value).await
    }
}

impl ModelSessionSender for Sender {
    fn send(&mut self, command: ModelSessionCommand) -> ModelSessionFuture<'_, ()> {
        Box::pin(async move {
            match command {
                ModelSessionCommand::Input(ModelInput::Audio { chunk }) => {
                    if chunk.format != audio_format(16000) || chunk.data.len() % 2 != 0 {
                        return Err(AgentError::Model(
                            "realtime input requires mono 16kHz PCM16".into(),
                        ));
                    }
                    self.write(json!({"type":"audio","audio":STANDARD.encode(chunk.data)}))
                        .await?;
                }
                ModelSessionCommand::Input(ModelInput::Text { text }) => {
                    if !self.capabilities.as_ref().is_some_and(|c| c.text_input) {
                        return Err(AgentError::Model(
                            "realtime bridge does not support text input".into(),
                        ));
                    }
                    self.write(json!({"type":"text","text":text})).await?;
                }
                ModelSessionCommand::Input(ModelInput::EndAudio) => {} // Server VAD commits speech.
                ModelSessionCommand::CancelResponse { response_id } => {
                    self.write(json!({"type":"cancel","response_id":response_id}))
                        .await?
                }
                ModelSessionCommand::Close => self.write(json!({"type":"close"})).await?,
                ModelSessionCommand::Observe { items } => {
                    for item in items {
                        match item.kind {
                            TurnItemKind::ToolCompleted {
                                call_id,
                                name,
                                result,
                                ..
                            }
                            | TurnItemKind::ToolOutput {
                                call_id,
                                name,
                                result,
                                ..
                            } => {
                                if !self.capabilities.as_ref().is_some_and(|c| c.tool_calls) {
                                    return Err(AgentError::Model(
                                        "realtime bridge does not support tool results".into(),
                                    ));
                                }
                                self.write(json!({"type":"tool_result","call_id":call_id,"name":name,"result":result})).await?;
                            }
                            TurnItemKind::ToolAccepted { .. }
                            | TurnItemKind::ToolProgress { .. } => {}
                            other => {
                                self.write(json!({"type":"observation","event":other}))
                                    .await?
                            }
                        }
                    }
                }
            }
            Ok(())
        })
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event {
    Ready {
        #[serde(default)]
        provider: Option<String>,
        #[serde(default)]
        audio: Option<AudioFormats>,
        #[serde(default)]
        capabilities: Option<Capabilities>,
    },
    Error {
        message: String,
    },
    InputStarted {
        item_id: String,
    },
    InputEnded {
        item_id: String,
    },
    InputCommitted {
        item_id: String,
    },
    TranscriptFinalized {
        item_id: String,
        text: String,
    },
    ResponseStarted {
        response_id: String,
    },
    TextDelta {
        response_id: String,
        item_id: String,
        text: String,
    },
    AudioDelta {
        response_id: String,
        item_id: String,
        audio: String,
        #[serde(default)]
        run_id: Option<String>,
        #[serde(default)]
        turn_id: Option<String>,
        #[serde(default)]
        chunk_id: Option<String>,
    },
    PlaybackStop {
        response_id: String,
        run_id: String,
        turn_id: String,
    },
    PlaybackAudioDone {
        run_id: String,
        turn_id: String,
    },
    ItemCompleted {
        response_id: String,
        item_id: String,
        text: Option<String>,
        function_calls: Vec<ModelFunctionCall>,
    },
    ResponseFinished {
        response_id: String,
        status: String,
        #[serde(default)]
        error: String,
    },
    Usage {
        event_id: String,
        response_id: String,
        usage: TokenUsage,
    },
}

impl Receiver {
    fn emit(&self, event: feedback::Event) {
        if let Some(sender) = &self.playback {
            let _ = sender.send(event);
        }
    }

    fn target(&self, run_id: String, turn_id: String) -> feedback::Target {
        feedback::Target {
            connection_id: self.connection_id.clone(),
            run_id,
            turn_id,
            writer: self.writer.clone(),
        }
    }
    async fn read(&mut self) -> Result<Option<Event>> {
        self.bridge.read().await
    }
}

impl ModelSessionReceiver for Receiver {
    fn receive(&mut self) -> ModelSessionFuture<'_, Option<ModelSessionEvent>> {
        Box::pin(async move {
            loop {
                let Some(event) = self.read().await? else {
                    return Ok(None);
                };
                return Ok(Some(match event {
                    Event::PlaybackStop {
                        response_id,
                        run_id,
                        turn_id,
                    } => {
                        self.emit(feedback::Event::Stop {
                            response_id,
                            target: self.target(run_id, turn_id),
                        });
                        continue;
                    }
                    Event::PlaybackAudioDone { run_id, turn_id } => {
                        self.emit(feedback::Event::AudioDone {
                            target: self.target(run_id, turn_id),
                        });
                        continue;
                    }
                    Event::Ready { .. } => {
                        return Err(AgentError::Model(
                            "unexpected duplicate worker ready".into(),
                        ));
                    }
                    Event::Error { message } => return Err(AgentError::Model(message)),
                    Event::InputStarted { item_id } => ModelSessionEvent::InputStarted { item_id },
                    Event::InputEnded { item_id } => ModelSessionEvent::InputEnded { item_id },
                    Event::InputCommitted { item_id } => {
                        ModelSessionEvent::InputCommitted { item_id }
                    }
                    Event::TranscriptFinalized { item_id, text } => {
                        ModelSessionEvent::TranscriptFinalized { item_id, text }
                    }
                    Event::ResponseStarted { response_id } => {
                        ModelSessionEvent::ResponseStarted { response_id }
                    }
                    Event::TextDelta {
                        response_id,
                        item_id,
                        text,
                    } => {
                        self.emit(feedback::Event::TextDelta {
                            response_id: response_id.clone(),
                            text: text.clone(),
                        });
                        ModelSessionEvent::TextDelta {
                            response_id,
                            item_id,
                            text,
                        }
                    }
                    Event::AudioDelta {
                        response_id,
                        item_id,
                        audio,
                        run_id,
                        turn_id,
                        chunk_id,
                    } => {
                        if self.playback_feedback
                            && (run_id.is_none() || turn_id.is_none() || chunk_id.is_none())
                        {
                            return Err(AgentError::Model(
                                "playback feedback requires complete chunk identity".into(),
                            ));
                        }
                        let chunk = AudioChunk {
                            format: audio_format(self.output_rate),
                            data: STANDARD.decode(audio).map_err(|_| {
                                AgentError::Model("invalid worker audio base64".into())
                            })?,
                        };
                        let target = match (run_id, turn_id) {
                            (Some(run), Some(turn)) => Some(self.target(run, turn)),
                            (None, None) => None,
                            _ => {
                                return Err(AgentError::Model(
                                    "incomplete playback context".into(),
                                ));
                            }
                        };
                        self.emit(feedback::Event::Audio {
                            response_id: response_id.clone(),
                            chunk_id: chunk_id.unwrap_or_else(|| item_id.clone()),
                            chunk: chunk.clone(),
                            target,
                        });
                        ModelSessionEvent::AudioDelta {
                            response_id,
                            item_id,
                            chunk,
                        }
                    }
                    Event::ItemCompleted {
                        response_id,
                        item_id,
                        text,
                        function_calls,
                    } => ModelSessionEvent::ItemCompleted {
                        response_id,
                        item_id,
                        text,
                        function_calls,
                    },
                    Event::ResponseFinished {
                        response_id,
                        status,
                        error,
                    } => {
                        let status = match status.as_str() {
                            "completed" => ModelResponseStatus::Completed,
                            "interrupted" => ModelResponseStatus::Interrupted { reason: error },
                            _ => ModelResponseStatus::Failed { error },
                        };
                        self.emit(feedback::Event::Finished {
                            response_id: response_id.clone(),
                        });
                        ModelSessionEvent::ResponseFinished {
                            response_id,
                            status,
                        }
                    }
                    Event::Usage {
                        event_id,
                        response_id,
                        usage,
                    } => ModelSessionEvent::Usage {
                        event_id,
                        response_id,
                        usage,
                    },
                }));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use canary_agent_kernel::{ModelRequest, Thread, ToolResult, TurnItem, TurnItemSource};

    #[tokio::test]
    async fn playback_lifecycle_round_trips_through_python_ipc() {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            for mode in ["complete", "stop"] {
                let (client, mut playback) = BridgeClient::new(BridgeConfig {
                    endpoint: "wss://example.invalid".into(),
                    api_key: "fake".into(),
                    model: mode.into(),
                    voice: "test".into(),
                    python: "python3".into(),
                    script: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("python/mock_playback_worker.py"),
                })
                .with_playback();
                let mut session = client
                    .connect(ModelSessionConfig {
                        connection_id: mode.into(),
                        turn_id: "turn".into(),
                        request: ModelRequest {
                            messages: vec![],
                            functions: vec![],
                        },
                        history: Thread::new("thread"),
                        model: client.model_descriptor(),
                    })
                    .await
                    .unwrap();
                assert!(matches!(
                    playback.recv().await,
                    Some(feedback::Event::Ready {
                        text_input: false,
                        ..
                    })
                ));
                session
                    .sender
                    .send(ModelSessionCommand::Input(ModelInput::Audio {
                        chunk: AudioChunk {
                            format: audio_format(16000),
                            data: vec![0; 640],
                        },
                    }))
                    .await
                    .unwrap();
                assert!(matches!(
                    session.receiver.receive().await.unwrap(),
                    Some(ModelSessionEvent::ResponseStarted { .. })
                ));
                assert!(matches!(
                    session.receiver.receive().await.unwrap(),
                    Some(ModelSessionEvent::AudioDelta { .. })
                ));
                assert!(matches!(
                    session.receiver.receive().await.unwrap(),
                    Some(ModelSessionEvent::ResponseFinished { .. })
                ));
                let Some(feedback::Event::Audio {
                    target: Some(target),
                    chunk_id,
                    chunk,
                    ..
                }) = playback.recv().await
                else {
                    panic!("missing tracked audio")
                };
                assert_eq!(chunk.format.sample_rate, 48000);
                let mut tracker = feedback::Tracker::default();
                tracker.enqueue(target.clone(), chunk_id, 0, 2);
                let now = std::time::Instant::now();
                let expected = if mode == "complete" {
                    assert!(matches!(
                        playback.recv().await,
                        Some(feedback::Event::AudioDone { .. })
                    ));
                    tracker.audio_done(target);
                    tracker.rendered(0, 2, now, 48000);
                    vec!["playback_chunk_completed", "playback_completed"]
                } else {
                    assert!(matches!(
                        playback.recv().await,
                        Some(feedback::Event::Stop { .. })
                    ));
                    tracker.stop_target(&target, now);
                    vec!["playback_stopped"]
                };
                let deliveries = tracker.drain(now + std::time::Duration::from_millis(1));
                assert_eq!(deliveries.len(), expected.len());
                for delivery in deliveries {
                    delivery.send().await.unwrap();
                }
                for kind in expected {
                    assert!(matches!(
                        session.receiver.receive().await.unwrap(),
                        Some(ModelSessionEvent::ResponseStarted { .. })
                    ));
                    let Some(ModelSessionEvent::TextDelta { text, .. }) =
                        session.receiver.receive().await.unwrap()
                    else {
                        panic!("missing feedback echo")
                    };
                    let message: Value = serde_json::from_str(&text).unwrap();
                    assert_eq!(message["type"], kind);
                    assert_eq!(message["run_id"], "run");
                    assert_eq!(message["turn_id"], "turn");
                    assert!(matches!(
                        session.receiver.receive().await.unwrap(),
                        Some(ModelSessionEvent::ResponseFinished { .. })
                    ));
                }
                session
                    .sender
                    .send(ModelSessionCommand::Close)
                    .await
                    .unwrap();
            }
        })
        .await
        .expect("offline bridge timed out");
    }

    #[tokio::test]
    async fn python_process_exchanges_audio_and_tool_events() {
        let client = BridgeClient {
            config: BridgeConfig {
                endpoint: "wss://example.com/realtime".into(),
                api_key: "fake".into(),
                model: "test".into(),
                voice: "Tina".into(),
                python: "python3".into(),
                script: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/mock_worker.py"),
            },
            worker: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/mock_worker.py"),
            playback: None,
        };
        let mut session = client
            .connect(ModelSessionConfig {
                connection_id: "conn".into(),
                turn_id: "turn".into(),
                request: ModelRequest {
                    messages: vec![],
                    functions: vec![],
                },
                history: Thread::new("thread"),
                model: client.model_descriptor(),
            })
            .await
            .unwrap();
        session
            .sender
            .send(ModelSessionCommand::Input(ModelInput::Audio {
                chunk: AudioChunk {
                    format: audio_format(16000),
                    data: vec![0, 0],
                },
            }))
            .await
            .unwrap();
        assert!(matches!(
            session.receiver.receive().await.unwrap(),
            Some(ModelSessionEvent::InputStarted { .. })
        ));
        assert!(matches!(
            session.receiver.receive().await.unwrap(),
            Some(ModelSessionEvent::TranscriptFinalized { .. })
        ));
        assert!(matches!(
            session.receiver.receive().await.unwrap(),
            Some(ModelSessionEvent::ResponseStarted { .. })
        ));
        assert!(
            matches!(session.receiver.receive().await.unwrap(), Some(ModelSessionEvent::ItemCompleted { function_calls, .. }) if function_calls[0].name == "get_current_time")
        );
        assert!(matches!(
            session.receiver.receive().await.unwrap(),
            Some(ModelSessionEvent::ResponseFinished { .. })
        ));
        session
            .sender
            .send(ModelSessionCommand::Observe {
                items: vec![TurnItem::new(
                    TurnItemSource::Tool,
                    TurnItemKind::ToolCompleted {
                        call_id: "call".into(),
                        name: "get_current_time".into(),
                        operation_id: "op".into(),
                        result: ToolResult::Success {
                            output: json!({"time":"now"}),
                        },
                    },
                )],
            })
            .await
            .unwrap();
        assert!(matches!(
            session.receiver.receive().await.unwrap(),
            Some(ModelSessionEvent::ResponseStarted { .. })
        ));
        assert!(
            matches!(session.receiver.receive().await.unwrap(), Some(ModelSessionEvent::AudioDelta { chunk, .. }) if chunk.data == [0, 0])
        );
        assert!(matches!(
            session.receiver.receive().await.unwrap(),
            Some(ModelSessionEvent::ResponseFinished { .. })
        ));
        session
            .sender
            .send(ModelSessionCommand::Input(ModelInput::Text {
                text: "typed input".into(),
            }))
            .await
            .unwrap();
        assert!(matches!(session.receiver.receive().await.unwrap(),
            Some(ModelSessionEvent::ResponseStarted { response_id }) if response_id == "text-response"));
        assert!(matches!(
            session.receiver.receive().await.unwrap(),
            Some(ModelSessionEvent::ResponseFinished { .. })
        ));
        session
            .sender
            .send(ModelSessionCommand::Close)
            .await
            .unwrap();
        assert!(session.receiver.receive().await.unwrap().is_none());
    }
}
