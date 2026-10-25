use base64::{Engine, engine::general_purpose::STANDARD};
use canary_agent_kernel::{ChatMessage, ModelFunctionCall, TokenUsage, TurnItemKind};
use canary_agent_runtime::{
    AgentError, AudioChunk, AudioEncoding, AudioFormat, ModelClient, ModelDescriptor, ModelInput,
    ModelInteraction, ModelResponseStatus, ModelSession, ModelSessionCommand, ModelSessionConfig,
    ModelSessionEvent, ModelSessionFuture, ModelSessionReceiver, ModelSessionSender, Result,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{path::PathBuf, process::Stdio, sync::Arc};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
};

#[derive(Clone)]
pub struct QwenConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub voice: String,
    pub python: String,
}

impl QwenConfig {
    pub fn validate(&mut self) -> Result<()> {
        if self.api_key.is_empty() || self.model.is_empty() {
            return Err(AgentError::Model(
                "realtime API key and model are required".into(),
            ));
        }
        let mut url = url::Url::parse(&self.endpoint)
            .map_err(|_| AgentError::Model("invalid Qwen endpoint".into()))?;
        let scheme = match url.scheme() {
            "https" | "wss" => "wss",
            "http" | "ws" => "ws",
            _ => {
                return Err(AgentError::Model(
                    "Qwen endpoint requires http(s) or ws(s)".into(),
                ));
            }
        };
        url.set_scheme(scheme)
            .map_err(|_| AgentError::Model("invalid Qwen URL scheme".into()))?;
        if !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(AgentError::Model("Qwen endpoint must not contain credentials, query, or fragment; set REAL_TIME_MODEL separately".into()));
        }
        if matches!(
            url.path().trim_end_matches('/'),
            "" | "/v1" | "/compatible-mode/v1"
        ) {
            url.set_path("/api-ws/v1/realtime");
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

pub struct QwenClient {
    config: QwenConfig,
    worker: PathBuf,
}

impl QwenClient {
    pub fn new(config: QwenConfig) -> Self {
        Self {
            config,
            worker: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/worker.py"),
        }
    }
}

// Both transport halves retain the process; the last drop terminates it.
struct Worker {
    _child: Child,
}
struct Sender {
    input: BufWriter<ChildStdin>,
    _worker: Arc<Worker>,
}
struct Receiver {
    output: Lines<BufReader<ChildStdout>>,
    _worker: Arc<Worker>,
}

impl ModelClient for QwenClient {
    fn model_descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            fqn: format!("dashscope/{}", self.config.model),
            settings: json!({"voice":self.config.voice,"transport":"python-sdk-ipc"}),
        }
    }
    fn interaction(&self) -> ModelInteraction {
        ModelInteraction::Bidirectional
    }
    fn connect(&self, config: ModelSessionConfig) -> ModelSessionFuture<'_, ModelSession> {
        Box::pin(async move {
            let mut child = Command::new(&self.config.python).arg("-u").arg(&self.worker)
                .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).kill_on_drop(true).spawn()
                .map_err(|e| AgentError::Model(format!("cannot start Qwen SDK worker: {e}; set VOICE_PYTHON to the Python environment with dashscope installed")))?;
            let input = child.stdin.take().expect("piped stdin");
            let output = child.stdout.take().expect("piped stdout");
            let worker = Arc::new(Worker { _child: child });
            let mut sender = Sender {
                input: BufWriter::new(input),
                _worker: worker.clone(),
            };
            let mut receiver = Receiver {
                output: BufReader::new(output).lines(),
                _worker: worker,
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
                .map_err(|_| AgentError::Model("Qwen SDK worker setup timed out".into()))??
            {
                Some(Event::Ready) => Ok(ModelSession {
                    sender: Box::new(sender),
                    receiver: Box::new(receiver),
                }),
                Some(Event::Error { message }) => Err(AgentError::Model(message)),
                _ => Err(AgentError::Model(
                    "Qwen SDK worker closed or sent an unexpected setup event".into(),
                )),
            }
        })
    }
}

impl Sender {
    async fn write(&mut self, value: Value) -> Result<()> {
        let mut data = serde_json::to_vec(&value)?;
        data.push(b'\n');
        self.input.write_all(&data).await?;
        self.input.flush().await?;
        Ok(())
    }
}

impl ModelSessionSender for Sender {
    fn send(&mut self, command: ModelSessionCommand) -> ModelSessionFuture<'_, ()> {
        Box::pin(async move {
            match command {
                ModelSessionCommand::Input(ModelInput::Audio { chunk }) => {
                    if chunk.format != audio_format(16000) || chunk.data.len() % 2 != 0 {
                        return Err(AgentError::Model(
                            "Qwen input requires mono 16kHz PCM16".into(),
                        ));
                    }
                    self.write(json!({"type":"audio","audio":STANDARD.encode(chunk.data)}))
                        .await?;
                }
                ModelSessionCommand::Input(ModelInput::Text { text }) => {
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
    Ready,
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
    async fn read(&mut self) -> Result<Option<Event>> {
        self.output
            .next_line()
            .await?
            .map(|line| {
                serde_json::from_str(&line)
                    .map_err(|e| AgentError::Model(format!("invalid Qwen worker event: {e}")))
            })
            .transpose()
    }
}

impl ModelSessionReceiver for Receiver {
    fn receive(&mut self) -> ModelSessionFuture<'_, Option<ModelSessionEvent>> {
        Box::pin(async move {
            let Some(event) = self.read().await? else {
                return Ok(None);
            };
            Ok(Some(match event {
                Event::Ready => {
                    return Err(AgentError::Model(
                        "unexpected duplicate worker ready".into(),
                    ));
                }
                Event::Error { message } => return Err(AgentError::Model(message)),
                Event::InputStarted { item_id } => ModelSessionEvent::InputStarted { item_id },
                Event::InputEnded { item_id } => ModelSessionEvent::InputEnded { item_id },
                Event::InputCommitted { item_id } => ModelSessionEvent::InputCommitted { item_id },
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
                } => ModelSessionEvent::TextDelta {
                    response_id,
                    item_id,
                    text,
                },
                Event::AudioDelta {
                    response_id,
                    item_id,
                    audio,
                } => ModelSessionEvent::AudioDelta {
                    response_id,
                    item_id,
                    chunk: AudioChunk {
                        format: audio_format(24000),
                        data: STANDARD
                            .decode(audio)
                            .map_err(|_| AgentError::Model("invalid worker audio base64".into()))?,
                    },
                },
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
                } => ModelSessionEvent::ResponseFinished {
                    response_id,
                    status: match status.as_str() {
                        "completed" => ModelResponseStatus::Completed,
                        "interrupted" => ModelResponseStatus::Interrupted { reason: error },
                        _ => ModelResponseStatus::Failed { error },
                    },
                },
                Event::Usage {
                    event_id,
                    response_id,
                    usage,
                } => ModelSessionEvent::Usage {
                    event_id,
                    response_id,
                    usage,
                },
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use canary_agent_kernel::{ModelRequest, Thread, ToolResult, TurnItem, TurnItemSource};

    #[tokio::test]
    async fn python_process_exchanges_audio_and_tool_events() {
        let client = QwenClient {
            config: QwenConfig {
                endpoint: "wss://example.com/realtime".into(),
                api_key: "fake".into(),
                model: "test".into(),
                voice: "Tina".into(),
                python: "python3".into(),
            },
            worker: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/mock_worker.py"),
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
