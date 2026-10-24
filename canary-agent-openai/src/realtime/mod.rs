mod config;
mod protocol;

use canary_agent_runtime::{
    AgentError, ModelClient, ModelDescriptor, ModelInteraction, ModelSession, ModelSessionCommand,
    ModelSessionConfig, ModelSessionEvent, ModelSessionFuture, ModelSessionReceiver,
    ModelSessionSender, Result,
};
pub use config::RealtimeConfig;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot, watch};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, http::HeaderValue, Message},
    MaybeTlsStream, WebSocketStream,
};

#[derive(Debug, Clone)]
pub struct RealtimeClient {
    config: RealtimeConfig,
}

impl RealtimeClient {
    pub fn new(config: RealtimeConfig) -> Self {
        Self { config }
    }
}

impl ModelClient for RealtimeClient {
    fn model_descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            fqn: format!("realtime/{}", self.config.model),
            settings: json!({
                "voice":self.config.voice,
                "transcription_model":self.config.transcription_model,
            }),
        }
    }
    fn interaction(&self) -> ModelInteraction {
        ModelInteraction::Bidirectional
    }
    fn connect(&self, config: ModelSessionConfig) -> ModelSessionFuture<'_, ModelSession> {
        Box::pin(async move {
            if self.config.api_key.is_empty() || self.config.model.is_empty() {
                return Err(AgentError::Model(
                    "realtime API key and model are required".into(),
                ));
            }
            let mut request = self
                .config
                .websocket_url()?
                .as_str()
                .into_client_request()
                .map_err(transport_error)?;
            let mut auth = HeaderValue::from_str(&format!("Bearer {}", self.config.api_key))
                .map_err(|_| AgentError::Model("invalid realtime API key header".into()))?;
            auth.set_sensitive(true);
            request.headers_mut().insert("Authorization", auth);
            let restore = protocol::restore(&config)?;
            let update = protocol::session_update(&self.config, &config.request);
            let connect = async {
                let (mut socket, _) = connect_async(request)
                    .await
                    .map_err(|error| setup_error("WebSocket handshake", error))?;
                send_json(&mut socket, update)
                    .await
                    .map_err(|error| setup_error("sending session.update", error))?;
                loop {
                    let raw = socket
                        .next()
                        .await
                        .ok_or_else(|| {
                            AgentError::Http("realtime connection closed during setup".into())
                        })?
                        .map_err(|error| setup_error("waiting for session.updated", error))?;
                    if let Message::Close(frame) = &raw {
                        return Err(setup_error(
                            "waiting for session.updated",
                            close_error(frame.as_ref()),
                        ));
                    }
                    if let Message::Text(text) = raw {
                        let value: Value = serde_json::from_str(text.as_ref())?;
                        protocol::check_error(&value)?;
                        if value["type"] == "session.updated" {
                            break;
                        }
                    }
                }
                for frame in &restore.frames {
                    send_json(&mut socket, frame.clone())
                        .await
                        .map_err(|error| setup_error("restoring conversation", error))?;
                }
                Ok::<_, AgentError>((socket, restore))
            };
            let (socket, restore) = tokio::time::timeout(self.config.connect_timeout, connect)
                .await
                .map_err(|_| AgentError::Http("realtime connection setup timed out".into()))??;
            let (sink, stream) = socket.split();
            let (wire, commands) = mpsc::channel(64);
            let (failure, failure_rx) = watch::channel(None);
            let task = tokio::spawn(write_socket(sink, commands, failure));
            let lifetime = Arc::new(WriterLifetime(task.abort_handle()));
            let requested = restore
                .frames
                .iter()
                .any(|frame| frame["type"] == "response.create");
            let initial_state = protocol::State::new(restore.waiting_calls, requested);
            let state = Arc::new(Mutex::new(initial_state));
            let sender = Sender {
                wire: wire.clone(),
                config: self.config.clone(),
                state: state.clone(),
                _lifetime: lifetime.clone(),
            };
            let receiver = Receiver {
                stream,
                wire,
                failure: failure_rx,
                config: self.config.clone(),
                state,
                pending: VecDeque::new(),
                item_text: std::collections::HashMap::new(),
                seen_items: std::collections::HashSet::new(),
                response_output: std::collections::HashMap::new(),
                _lifetime: lifetime,
            };
            Ok(ModelSession {
                sender: Box::new(sender),
                receiver: Box::new(receiver),
            })
        })
    }
}

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;
struct Wire {
    frames: Vec<Value>,
    reply: Option<oneshot::Sender<Result<()>>>,
    close: bool,
    control: Option<Message>,
}
struct WriterLifetime(tokio::task::AbortHandle);
impl Drop for WriterLifetime {
    fn drop(&mut self) {
        self.0.abort();
    }
}
struct Sender {
    wire: mpsc::Sender<Wire>,
    config: RealtimeConfig,
    state: Arc<Mutex<protocol::State>>,
    _lifetime: Arc<WriterLifetime>,
}
struct Receiver {
    stream: futures_util::stream::SplitStream<Socket>,
    wire: mpsc::Sender<Wire>,
    failure: watch::Receiver<Option<String>>,
    config: RealtimeConfig,
    state: Arc<Mutex<protocol::State>>,
    pending: VecDeque<ModelSessionEvent>,
    item_text: std::collections::HashMap<String, String>,
    seen_items: std::collections::HashSet<String>,
    response_output: std::collections::HashMap<String, protocol::ResponseOutput>,
    _lifetime: Arc<WriterLifetime>,
}

impl ModelSessionSender for Sender {
    fn send(&mut self, command: ModelSessionCommand) -> ModelSessionFuture<'_, ()> {
        Box::pin(async move {
            let close = matches!(command, ModelSessionCommand::Close);
            let permit = self
                .wire
                .reserve()
                .await
                .map_err(|_| AgentError::Http("realtime writer closed".into()))?;
            let result = {
                let mut state = self
                    .state
                    .lock()
                    .map_err(|_| AgentError::Model("realtime state poisoned".into()))?;
                let frames = protocol::command(&self.config, &mut state, command)?;
                if frames.is_empty() && !close {
                    return Ok(());
                }
                let (reply, result) = oneshot::channel();
                // Commit protocol state and queue its frames together, so the
                // reader cannot enqueue a continuation ahead of a tool result.
                permit.send(Wire {
                    frames,
                    reply: Some(reply),
                    close,
                    control: None,
                });
                result
            };
            result
                .await
                .map_err(|_| AgentError::Http("realtime writer closed".into()))?
        })
    }
}

impl ModelSessionReceiver for Receiver {
    fn receive(&mut self) -> ModelSessionFuture<'_, Option<ModelSessionEvent>> {
        Box::pin(async move {
            loop {
                if let Some(event) = self.pending.pop_front() {
                    return Ok(Some(event));
                }
                if let Some(error) = self.failure.borrow().clone() {
                    return Err(AgentError::Http(error));
                }
                let frame = tokio::select! {
                    changed = self.failure.changed() => {
                        if changed.is_err() { return Ok(None); }
                        continue;
                    }
                    frame = self.stream.next() => frame,
                };
                match frame {
                    Some(Ok(Message::Text(text))) => {
                        let value: Value =
                            serde_json::from_str(text.as_ref()).map_err(|error| {
                                AgentError::Model(format!(
                                    "invalid realtime event JSON ({} bytes): {error}",
                                    text.len()
                                ))
                            })?;
                        self.parse(value)?;
                    }
                    Some(Ok(Message::Close(frame))) => return Err(close_error(frame.as_ref())),
                    None => return Ok(None),
                    Some(Ok(Message::Ping(data))) => self
                        .wire
                        .try_send(Wire {
                            frames: vec![],
                            reply: None,
                            close: false,
                            control: Some(Message::Pong(data)),
                        })
                        .map_err(|_| {
                            AgentError::Http("realtime control queue unavailable".into())
                        })?,
                    Some(Err(error)) => return Err(transport_error(error)),
                    _ => {}
                }
            }
        })
    }
}

async fn send_json(socket: &mut Socket, value: Value) -> Result<()> {
    socket
        .send(Message::Text(value.to_string().into()))
        .await
        .map_err(transport_error)
}

async fn write_socket(
    mut sink: futures_util::stream::SplitSink<Socket, Message>,
    mut commands: mpsc::Receiver<Wire>,
    failure: watch::Sender<Option<String>>,
) {
    while let Some(command) = commands.recv().await {
        let result = async {
            if let Some(control) = command.control {
                sink.send(control).await.map_err(transport_error)?;
            }
            for frame in command.frames {
                sink.send(Message::Text(frame.to_string().into()))
                    .await
                    .map_err(transport_error)?;
            }
            if command.close {
                sink.close().await.map_err(transport_error)?;
            }
            Ok(())
        }
        .await;
        let failed = result.as_ref().err().map(ToString::to_string);
        if let Some(reply) = command.reply {
            let _ = reply.send(result);
        }
        if let Some(error) = failed {
            let _ = failure.send(Some(error));
            return;
        }
        if command.close {
            return;
        }
    }
    let _ = sink.close().await;
}

fn transport_error(error: impl std::fmt::Display) -> AgentError {
    AgentError::Http(format!("realtime transport: {error}"))
}

fn setup_error(phase: &str, error: impl std::fmt::Display) -> AgentError {
    AgentError::Http(format!("realtime setup ({phase}): {error}"))
}

fn close_error(frame: Option<&tokio_tungstenite::tungstenite::protocol::CloseFrame>) -> AgentError {
    match frame {
        Some(frame) => AgentError::Http(format!(
            "realtime peer closed WebSocket: code={}, reason={}",
            frame.code, frame.reason
        )),
        None => AgentError::Http("realtime peer closed WebSocket without a reason".into()),
    }
}

#[cfg(test)]
mod tests;
