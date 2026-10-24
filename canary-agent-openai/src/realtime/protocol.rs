use super::*;
use base64::{engine::general_purpose::STANDARD, Engine};
use canary_agent_kernel::{new_id, ChatMessage, TokenUsage, ToolResult, TurnItemKind};
use canary_agent_runtime::{AudioChunk, ModelFunctionCall, ModelInput, ModelResponseStatus};
use std::collections::{BTreeSet, HashMap, HashSet};

pub(super) struct State {
    active: BTreeSet<String>,
    interrupted: HashSet<String>,
    waiting: BTreeSet<String>,
    completed_calls: HashSet<String>,
    text_inputs: HashMap<String, String>,
    needs_response: bool,
    speaking: bool,
    closing: bool,
    requested: bool,
    awaiting_vad: bool,
}

impl State {
    pub(super) fn new(waiting: BTreeSet<String>, requested: bool) -> Self {
        Self {
            active: BTreeSet::new(),
            interrupted: HashSet::new(),
            waiting,
            completed_calls: HashSet::new(),
            text_inputs: HashMap::new(),
            needs_response: false,
            speaking: false,
            closing: false,
            requested,
            awaiting_vad: false,
        }
    }
    fn continuation(&mut self) -> Option<Value> {
        if self.needs_response
            && self.active.is_empty()
            && !self.speaking
            && self.waiting.is_empty()
            && !self.closing
            && !self.requested
            && !self.awaiting_vad
        {
            self.needs_response = false;
            self.requested = true;
            Some(json!({"type":"response.create"}))
        } else {
            None
        }
    }
}

pub(super) fn session_update(
    config: &RealtimeConfig,
    request: &canary_agent_runtime::ModelRequest,
) -> Value {
    let instructions = request
        .messages
        .iter()
        .filter_map(|m| match m {
            ChatMessage::System { content } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let tools: Vec<_> = request.functions.iter().map(|f| json!({
        "type":"function", "name":f.name, "description":f.description, "parameters":f.parameters
    })).collect();
    let vad = json!({"type":"server_vad","threshold":0.5,"silence_duration_ms":600,"interrupt_response":true,"create_response":true});
    let session = json!({"type":"realtime","output_modalities":["audio"],"instructions":instructions,"tools":tools,
        "audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":config.transcription_model},"turn_detection":vad},
        "output":{"format":{"type":"audio/pcm","rate":24000},"voice":config.voice}}});
    json!({"type":"session.update","session":session})
}

pub(super) struct Restoration {
    pub frames: Vec<Value>,
    pub waiting_calls: BTreeSet<String>,
}

pub(super) fn restore(config: &ModelSessionConfig) -> Result<Restoration> {
    let mut results = HashMap::new();
    for turn in &config.history.turns {
        for item in &turn.items {
            if let TurnItemKind::ToolCompleted {
                call_id, result, ..
            } = &item.kind
            {
                results.insert(call_id.as_str(), result);
            }
        }
    }
    let mut frames = Vec::new();
    let mut waiting = BTreeSet::new();
    let mut answered = HashSet::new();
    for message in &config.request.messages {
        match message {
            ChatMessage::System { .. } => {}
            ChatMessage::User { content } => frames.push(message_frame("user", content)),
            ChatMessage::Assistant {
                content,
                tool_calls,
                ..
            } => {
                if let Some(text) = content.as_ref().filter(|t| !t.is_empty()) {
                    frames.push(message_frame("assistant", text));
                }
                for call in tool_calls {
                    frames.push(json!({"type":"conversation.item.create","item":{"type":"function_call","call_id":call.call_id,"name":call.name,"arguments":call.arguments.to_string()}}));
                    waiting.insert(call.call_id.clone());
                }
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
                ..
            } => {
                let output = if content.get("status").and_then(Value::as_str) == Some("accepted") {
                    results
                        .get(tool_call_id.as_str())
                        .map(serde_json::to_string)
                        .transpose()?
                } else {
                    Some(content.to_string())
                };
                if let Some(output) = output {
                    if answered.insert(tool_call_id.clone()) {
                        frames.push(output_frame(tool_call_id, output));
                        waiting.remove(tool_call_id);
                    }
                }
            }
        }
    }
    if !frames.is_empty() && waiting.is_empty() {
        frames.push(json!({"type":"response.create"}));
    }
    Ok(Restoration {
        frames,
        waiting_calls: waiting,
    })
}

pub(super) fn command(
    config: &RealtimeConfig,
    state: &mut State,
    command: ModelSessionCommand,
) -> Result<Vec<Value>> {
    let mut frames = Vec::new();
    match command {
        ModelSessionCommand::Input(ModelInput::Audio { chunk }) => {
            if chunk.format != config.input_format() || chunk.data.len() % 2 != 0 {
                return Err(AgentError::Model(
                    "realtime input must match the configured mono PCM16 sample rate".into(),
                ));
            }
            frames.push(
                json!({"type":"input_audio_buffer.append","audio":STANDARD.encode(chunk.data)}),
            );
        }
        ModelSessionCommand::Input(ModelInput::Text { text }) => {
            let id = new_id("text");
            state.text_inputs.insert(id.clone(), text.clone());
            let mut frame = message_frame("user", &text);
            frame["item"]["id"] = json!(id);
            frames.push(frame);
            state.needs_response = true;
        }
        ModelSessionCommand::Input(ModelInput::EndAudio) => {
            frames.push(json!({"type":"input_audio_buffer.commit"}));
            state.needs_response = true;
        }
        ModelSessionCommand::CancelResponse { response_id } => {
            if state.active.contains(&response_id) {
                frames.push(json!({"type":"response.cancel","response_id":response_id}));
            }
        }
        ModelSessionCommand::Close => {
            state.closing = true;
        }
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
                        if !state.completed_calls.insert(call_id.clone()) {
                            continue;
                        }
                        let closed = name == "end_turn"
                            && matches!(&result, ToolResult::Success { output } if output.get("status").and_then(Value::as_str) == Some("closed"));
                        frames.push(output_frame(&call_id, serde_json::to_string(&result)?));
                        state.waiting.remove(&call_id);
                        if closed {
                            state.closing = true;
                        } else {
                            state.needs_response = true;
                        }
                    }
                    // Acceptance is not a native function result. Leave the call
                    // pending while the service continues receiving user audio.
                    TurnItemKind::ToolAccepted { .. } | TurnItemKind::ToolProgress { .. } => {}
                    other => frames.push(message_frame(
                        "user",
                        &json!({"source":"runtime","event":other}).to_string(),
                    )),
                }
            }
        }
    }
    if let Some(frame) = state.continuation() {
        frames.push(frame);
    }
    Ok(frames)
}

fn message_frame(role: &str, text: &str) -> Value {
    json!({"type":"conversation.item.create","item":{"type":"message","role":role,"content":[{"type":if role == "assistant" { "output_text" } else { "input_text" },"text":text}]}})
}
fn output_frame(id: &str, output: String) -> Value {
    json!({"type":"conversation.item.create","item":{"type":"function_call_output","call_id":id,"output":output}})
}
pub(super) fn check_error(value: &Value) -> Result<()> {
    if value["type"] == "error" || value.get("error").is_some_and(|error| !error.is_null()) {
        return Err(AgentError::Model(format!(
            "realtime provider error: {}",
            value["error"]
        )));
    }
    Ok(())
}
fn field<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value
        .get(name)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| AgentError::Model(format!("realtime event missing {name}")))
}

fn text_field<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| AgentError::Model(format!("realtime event missing {name}")))
}

#[derive(Default)]
pub(super) struct ResponseOutput {
    items: Vec<Value>,
    arguments: HashMap<String, FunctionArguments>,
}

#[derive(Default)]
struct FunctionArguments {
    deltas: String,
    done: Option<String>,
}

impl Receiver {
    pub(super) fn parse(&mut self, value: Value) -> Result<()> {
        check_error(&value)?;
        let kind = field(&value, "type")?;
        let shared = self.state.clone();
        let mut state = shared
            .lock()
            .map_err(|_| AgentError::Model("realtime state poisoned".into()))?;
        match kind {
            "input_audio_buffer.speech_started" => {
                state.speaking = true;
                state.awaiting_vad = true;
                self.pending.push_back(ModelSessionEvent::InputStarted {
                    item_id: field(&value, "item_id")?.into(),
                });
                // Quarantine output superseded by a server-detected interruption.
                for id in state.active.clone() {
                    if state.interrupted.insert(id.clone()) {
                        self.pending.push_back(ModelSessionEvent::ResponseFinished {
                            response_id: id,
                            status: ModelResponseStatus::Interrupted {
                                reason: "server detected new speech".into(),
                            },
                        });
                    }
                }
            }
            "input_audio_buffer.speech_stopped" => {
                self.pending.push_back(ModelSessionEvent::InputEnded {
                    item_id: field(&value, "item_id")?.into(),
                })
            }
            "input_audio_buffer.committed" => {
                state.speaking = false;
                self.pending.push_back(ModelSessionEvent::InputCommitted {
                    item_id: field(&value, "item_id")?.into(),
                });
            }
            "conversation.item.input_audio_transcription.completed" => {
                self.pending
                    .push_back(ModelSessionEvent::TranscriptFinalized {
                        item_id: field(&value, "item_id")?.into(),
                        text: text_field(&value, "transcript")?.into(),
                    })
            }
            "conversation.item.input_audio_transcription.failed" => {
                return Err(AgentError::Model(format!(
                    "input transcription failed: {}",
                    value["error"]
                )))
            }
            "conversation.item.created" => {
                if let Some(id) = value["item"]["id"].as_str() {
                    if let Some(text) = state.text_inputs.remove(id) {
                        self.pending
                            .push_back(ModelSessionEvent::InputCommitted { item_id: id.into() });
                        self.pending
                            .push_back(ModelSessionEvent::TranscriptFinalized {
                                item_id: id.into(),
                                text,
                            });
                    }
                }
            }
            "response.created" => {
                state.awaiting_vad = false;
                state.requested = false;
                state.needs_response = false;
                let id = field(&value["response"], "id")?.to_owned();
                state.active.insert(id.clone());
                self.pending
                    .push_back(ModelSessionEvent::ResponseStarted { response_id: id });
            }
            "response.audio.delta" | "response.output_audio.delta" => {
                let delta = text_field(&value, "delta")?;
                if delta.is_empty() {
                    return Ok(());
                }
                let data = STANDARD
                    .decode(delta)
                    .map_err(|_| AgentError::Model("invalid realtime audio base64".into()))?;
                if data.len() % 2 != 0 {
                    return Err(AgentError::Model("unaligned realtime PCM16 output".into()));
                }
                self.pending.push_back(ModelSessionEvent::AudioDelta {
                    response_id: field(&value, "response_id")?.into(),
                    item_id: field(&value, "item_id")?.into(),
                    chunk: AudioChunk {
                        format: self.config.output_format(),
                        data,
                    },
                });
            }
            "response.text.delta"
            | "response.output_text.delta"
            | "response.audio_transcript.delta"
            | "response.output_audio_transcript.delta" => {
                let id = field(&value, "item_id")?.to_owned();
                let text = text_field(&value, "delta")?.to_owned();
                if text.is_empty() {
                    return Ok(());
                }
                self.item_text
                    .entry(id.clone())
                    .or_default()
                    .push_str(&text);
                self.pending.push_back(ModelSessionEvent::TextDelta {
                    response_id: field(&value, "response_id")?.into(),
                    item_id: id,
                    text,
                });
            }
            "response.output_item.done" => {
                let response = field(&value, "response_id")?;
                if !state.interrupted.contains(response) {
                    if value["item"]["type"] == "function_call" {
                        self.response_output
                            .entry(response.into())
                            .or_default()
                            .items
                            .push(value["item"].clone());
                    } else {
                        self.complete_item(response, &value["item"], &mut state, None, None)?;
                    }
                }
            }
            "response.function_call_arguments.delta" | "response.function_call_arguments.done" => {
                let response = field(&value, "response_id")?;
                if !state.interrupted.contains(response) {
                    let arguments = self
                        .response_output
                        .entry(response.into())
                        .or_default()
                        .arguments
                        .entry(field(&value, "item_id")?.into())
                        .or_default();
                    if kind.ends_with(".delta") {
                        arguments.deltas.push_str(text_field(&value, "delta")?);
                    } else {
                        arguments.done = Some(text_field(&value, "arguments")?.into());
                    }
                }
            }
            "response.done" => {
                let response = &value["response"];
                let id = field(response, "id")?;
                let buffered = self.response_output.remove(id).unwrap_or_default();
                // A done item can contain truncated arguments when its response
                // is cancelled or incomplete. Only successful responses issue tools.
                if !state.interrupted.contains(id) && response["status"] == "completed" {
                    let mut items = response["output"].as_array().cloned().unwrap_or_default();
                    for item in &buffered.items {
                        if !items.iter().any(|other| other["id"] == item["id"]) {
                            items.push(item.clone());
                        }
                    }
                    for item in items {
                        let item_id = field(&item, "id")?;
                        let snapshot = buffered.items.iter().find(|i| i["id"] == item["id"]);
                        self.complete_item(
                            id,
                            &item,
                            &mut state,
                            buffered.arguments.get(item_id),
                            snapshot,
                        )?;
                    }
                }
                if let Some(usage) = response.get("usage").filter(|u| u.is_object()) {
                    let input = usage["input_tokens"].as_u64().unwrap_or(0);
                    let output = usage["output_tokens"].as_u64().unwrap_or(0);
                    self.pending.push_back(ModelSessionEvent::Usage {
                        event_id: value["event_id"]
                            .as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("usage_{id}")),
                        response_id: id.into(),
                        usage: TokenUsage {
                            input_tokens: input,
                            output_tokens: output,
                            cached_input_tokens: usage["input_token_details"]["cached_tokens"]
                                .as_u64()
                                .or_else(|| usage["input_tokens_details"]["cached_tokens"].as_u64())
                                .unwrap_or(0),
                            total_tokens: usage["total_tokens"].as_u64().unwrap_or(input + output),
                        },
                    });
                }
                state.active.remove(id);
                if !state.interrupted.contains(id) {
                    let status = match response["status"].as_str() {
                        Some("completed") => ModelResponseStatus::Completed,
                        Some("cancelled") | Some("canceled") => ModelResponseStatus::Interrupted {
                            reason: "provider cancelled response".into(),
                        },
                        _ => ModelResponseStatus::Failed {
                            error: format!(
                                "realtime response failed or incomplete: {}",
                                response["status_details"]
                            ),
                        },
                    };
                    self.pending.push_back(ModelSessionEvent::ResponseFinished {
                        response_id: id.into(),
                        status,
                    });
                }
            }
            _ => {}
        }
        if let Some(frame) = state.continuation() {
            self.wire
                .try_send(Wire {
                    frames: vec![frame],
                    reply: None,
                    close: false,
                    control: None,
                })
                .map_err(|_| AgentError::Http("realtime control queue unavailable".into()))?;
        }
        Ok(())
    }

    fn complete_item(
        &mut self,
        response: &str,
        item: &Value,
        state: &mut State,
        cached: Option<&FunctionArguments>,
        snapshot: Option<&Value>,
    ) -> Result<()> {
        let id = field(item, "id")?;
        if state.interrupted.contains(response) || self.seen_items.contains(id) {
            return Ok(());
        }
        let mut calls = Vec::new();
        if item["type"] == "function_call" {
            let call_id = field(item, "call_id")?.to_owned();
            if item["status"]
                .as_str()
                .is_some_and(|status| status != "completed")
            {
                return Err(AgentError::Model(format!("incomplete realtime function item: response={response}, item={id}, call={call_id}")));
            }
            let candidates = [
                cached.and_then(|a| a.done.as_deref()),
                item["arguments"].as_str(),
                cached.map(|a| a.deltas.as_str()),
                snapshot.and_then(|i| i["arguments"].as_str()),
            ];
            let arguments = candidates.into_iter().flatten()
                .filter_map(|raw| serde_json::from_str::<Value>(raw).ok())
                .find(Value::is_object)
                .ok_or_else(|| AgentError::Model(format!(
                    "invalid completed realtime function arguments: response={response}, item={id}, call={call_id}; expected a complete JSON object"
                )))?;
            calls.push(ModelFunctionCall {
                call_id: call_id.clone(),
                name: field(item, "name")?.into(),
                arguments,
            });
            state.waiting.insert(call_id);
        }
        self.seen_items.insert(id.into());
        let text = item["content"]
            .as_array()
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p["transcript"].as_str().or_else(|| p["text"].as_str()))
                    .collect::<Vec<_>>()
                    .join("")
            })
            .filter(|s| !s.is_empty())
            .or_else(|| self.item_text.remove(id));
        self.pending.push_back(ModelSessionEvent::ItemCompleted {
            response_id: response.into(),
            item_id: id.into(),
            text,
            function_calls: calls,
        });
        Ok(())
    }
}
