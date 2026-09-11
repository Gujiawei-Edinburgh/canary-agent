// Portions adapted from OpenAI Codex (Apache-2.0), commit fc948f8c47.
// Copyright 2025 OpenAI. See third_party/codex/NOTICE and LICENSE.
// Adaptations: Canary callbacks/metrics, local continuation persistence,
// completed-only provider fallback, and fail-closed malformed-event handling.
use crate::config::{ModelConfig, RetryConfig};
use crate::transport::send_with_retries;
use canary_agent_kernel::{
    ChatMessage, ModelContinuation, ModelFunctionCall, ModelRequest, ModelResponse,
    ModelStreamEvent, TokenUsage,
};
use canary_agent_runtime::model::{ModelClient, ModelDescriptor, ModelStreamHandler};
use canary_agent_runtime::{AgentError, Result};
use eventsource_stream::Eventsource;
use futures_util::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

const CONTINUATION_FORMAT: &str = "openai.responses.v1";
const OUTPUT_TEXT_DELTA: &str = "response.output_text.delta";
const REFUSAL_DELTA: &str = "response.refusal.delta";
const FUNCTION_ARGUMENTS_DELTA: &str = "response.function_call_arguments.delta";
const FUNCTION_ARGUMENTS_DONE: &str = "response.function_call_arguments.done";
const OUTPUT_ITEM_ADDED: &str = "response.output_item.added";
const OUTPUT_ITEM_DONE: &str = "response.output_item.done";
const RESPONSE_COMPLETED: &str = "response.completed";
const RESPONSE_FAILED: &str = "response.failed";
const RESPONSE_INCOMPLETE: &str = "response.incomplete";
const ERROR: &str = "error";

/// Stateless Responses API client. Continuation items travel through model history,
/// rather than an in-memory cache or a server-side conversation.
#[derive(Debug, Clone)]
pub struct ResponsesClient {
    http: reqwest::Client,
    config: ModelConfig,
    retry: RetryConfig,
    stream_idle_timeout: Duration,
}

impl ResponsesClient {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            config,
            retry: RetryConfig::default(),
            stream_idle_timeout: Duration::from_secs(300),
        }
    }

    pub fn with_retry_config(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
        self
    }

    /// Maximum wait for the next complete SSE event (including lifecycle events).
    pub fn with_stream_idle_timeout(mut self, timeout: Duration) -> Self {
        self.stream_idle_timeout = timeout;
        self
    }
}

impl ModelClient for ResponsesClient {
    fn model_descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            fqn: self.config.model.clone(),
            settings: json!({
                "api": "responses",
                "reasoning_effort": self.config.reasoning_effort,
            }),
        }
    }

    fn stream_complete<'a>(
        &'a self,
        request: ModelRequest,
        on_event: &'a mut ModelStreamHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
        Box::pin(async move {
            let body = request_body(&self.config, request)?;
            let url = format!("{}/responses", self.config.base_url.trim_end_matches('/'));
            let response =
                send_with_retries(&self.http, &self.config, &self.retry, &url, &body).await?;
            read_response_stream(response.bytes_stream(), self.stream_idle_timeout, on_event).await
        })
    }
}

// Port of Codex process_sse's eventsource/timeout/completion loop. The owning
// request future provides cancellation; no detached task or Codex channel is needed.
async fn read_response_stream<S, B, E>(
    stream: S,
    idle_timeout: Duration,
    emit: &mut ModelStreamHandler<'_>,
) -> Result<ModelResponse>
where
    S: Stream<Item = std::result::Result<B, E>>,
    B: AsRef<[u8]>,
    E: std::fmt::Display,
{
    let stream = stream.eventsource();
    futures_util::pin_mut!(stream);
    let mut decoder = ResponseDecoder::default();
    loop {
        let next = tokio::time::timeout(idle_timeout, stream.next())
            .await
            .map_err(|_| {
                AgentError::Model("idle timeout waiting for Responses SSE event".into())
            })?;
        match next {
            Some(Ok(event)) => {
                decoder.data(&event.data, emit)?;
                if let Some(response) = decoder.completed.take() {
                    return Ok(response);
                }
            }
            Some(Err(error)) => {
                return Err(AgentError::Model(format!("Responses SSE error: {error}")))
            }
            None => return decoder.finish(),
        }
    }
}

// Adapted subset of Codex ResponsesStreamEvent. Provider output items remain
// opaque so encrypted reasoning and additional item fields survive persistence.
#[derive(Debug, Deserialize)]
struct ResponsesStreamEvent {
    #[serde(rename = "type")]
    kind: String,
    response: Option<Value>,
    item: Option<Value>,
    item_id: Option<String>,
    output_index: Option<u64>,
    delta: Option<String>,
    arguments: Option<String>,
    code: Option<String>,
    message: Option<String>,
}

fn request_body(config: &ModelConfig, request: ModelRequest) -> Result<Value> {
    let mut input = Vec::new();
    for message in request.messages {
        match message {
            ChatMessage::System { content } => {
                input.push(json!({"role": "system", "content": content}))
            }
            ChatMessage::User { content } => {
                input.push(json!({"role": "user", "content": content}))
            }
            ChatMessage::Assistant {
                content,
                tool_calls,
                continuation,
            } => {
                if let Some(continuation) = continuation {
                    if continuation.format != CONTINUATION_FORMAT {
                        return Err(AgentError::Model(format!(
                            "unsupported Responses continuation format: {}",
                            continuation.format
                        )));
                    }
                    let items = continuation.payload.as_array().ok_or_else(|| {
                        AgentError::Model(
                            "Responses continuation must contain an output-item array".into(),
                        )
                    })?;
                    // The payload is the complete output sequence for this response;
                    // replay it instead of also reconstructing text and tool calls.
                    input.extend(items.iter().cloned());
                } else {
                    if let Some(text) = content {
                        input.push(json!({
                            "type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": text, "annotations": []}]
                        }));
                    }
                    for call in tool_calls {
                        input.push(json!({
                            "type": "function_call", "call_id": call.call_id,
                            "name": call.name, "arguments": call.arguments.to_string()
                        }));
                    }
                }
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
                ..
            } => {
                input.push(json!({
                    "type": "function_call_output", "call_id": tool_call_id,
                    "output": content.as_str().map(str::to_owned).unwrap_or_else(|| content.to_string())
                }));
            }
        }
    }
    let tools: Vec<_> = request
        .functions
        .into_iter()
        .map(|function| {
            json!({
                "type": "function", "name": function.name, "description": function.description,
                "parameters": function.parameters, "strict": false
            })
        })
        .collect();
    let mut body = json!({
        "model": config.model, "input": input, "stream": true, "store": false,
        "include": ["reasoning.encrypted_content"]
    });
    if !config.reasoning_effort.is_empty() {
        body["reasoning"] = json!({"effort": config.reasoning_effort});
    }
    if !tools.is_empty() {
        body["tools"] = json!(tools);
        body["tool_choice"] = json!("auto");
    }
    Ok(body)
}

#[derive(Default)]
struct ResponseDecoder {
    completed: Option<ModelResponse>,
    added_items: BTreeMap<u64, Value>,
    done_items: BTreeMap<u64, Value>,
}

impl ResponseDecoder {
    #[cfg(test)]
    fn frame(&mut self, frame: &str, emit: &mut ModelStreamHandler<'_>) -> Result<()> {
        let data = frame
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(|line| line.strip_prefix(' ').unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n");
        self.data(&data, emit)
    }

    fn data(&mut self, data: &str, emit: &mut ModelStreamHandler<'_>) -> Result<()> {
        if data.trim().is_empty() || data.trim() == "[DONE]" {
            return Ok(());
        }
        let event: ResponsesStreamEvent = serde_json::from_str(data)
            .map_err(|e| AgentError::Model(format!("invalid Responses SSE event: {e}")))?;
        match event.kind.as_str() {
            OUTPUT_TEXT_DELTA | REFUSAL_DELTA => {
                let delta = event.delta.ok_or_else(|| {
                    AgentError::Model("Responses text event missing delta".into())
                })?;
                if !delta.is_empty() {
                    emit(ModelStreamEvent::OutputProgress);
                    emit(ModelStreamEvent::AssistantDelta { text: delta });
                }
            }
            FUNCTION_ARGUMENTS_DELTA => {
                let delta = event.delta.ok_or_else(|| {
                    AgentError::Model("Responses argument event missing delta".into())
                })?;
                if !delta.is_empty() {
                    emit(ModelStreamEvent::OutputProgress);
                }
            }
            OUTPUT_ITEM_ADDED => {
                let item = event.item.ok_or_else(|| {
                    AgentError::Model("Responses output_item.added missing item".into())
                })?;
                let index = self.item_index(event.output_index, &item);
                self.added_items.insert(index, item.clone());
                if item["type"] == "function_call"
                    && (item["name"].as_str().is_some_and(|s| !s.is_empty())
                        || item["call_id"].as_str().is_some_and(|s| !s.is_empty()))
                {
                    emit(ModelStreamEvent::OutputProgress);
                }
            }
            OUTPUT_ITEM_DONE => {
                let item = event.item.ok_or_else(|| {
                    AgentError::Model("Responses output_item.done missing item".into())
                })?;
                let index = self.item_index(event.output_index, &item);
                if let Some(added) = self.added_items.get(&index) {
                    check_item_identity(added, &item)?;
                }
                self.done_items.insert(index, item);
            }
            FUNCTION_ARGUMENTS_DONE => {
                // Compatibility with providers that send arguments.done but omit
                // output_item.done. Never concatenate a done snapshot onto deltas.
                let index = event.output_index.or_else(|| {
                    self.added_items
                        .iter()
                        .find(|(_, item)| {
                            event.item_id.as_deref().is_some_and(|id| item["id"] == id)
                        })
                        .map(|(index, _)| *index)
                });
                if let Some(item) = index.and_then(|index| self.added_items.get(&index)) {
                    if item["type"] == "function_call" {
                        if let (Some(expected), Some(actual)) =
                            (item["id"].as_str(), event.item_id.as_deref())
                        {
                            if expected != actual {
                                return Err(AgentError::Model(
                                    "Responses argument item_id mismatch".into(),
                                ));
                            }
                        }
                        let mut done = item.clone();
                        done["arguments"] =
                            json!(event.arguments.ok_or_else(|| AgentError::Model(
                                "Responses arguments.done missing arguments".into()
                            ))?);
                        done["status"] = json!("completed");
                        self.done_items.insert(index.unwrap(), done);
                    }
                }
            }
            RESPONSE_COMPLETED | RESPONSE_FAILED | RESPONSE_INCOMPLETE => {
                let response = event.response.ok_or_else(|| {
                    AgentError::Model("Responses terminal event missing response".into())
                })?;
                if let Some(usage) = response.get("usage").filter(|v| !v.is_null()) {
                    emit(ModelStreamEvent::TokenUsage {
                        usage: TokenUsage {
                            input_tokens: usage["input_tokens"].as_u64().unwrap_or_default(),
                            cached_input_tokens: usage["input_tokens_details"]["cached_tokens"]
                                .as_u64()
                                .unwrap_or_default(),
                            output_tokens: usage["output_tokens"].as_u64().unwrap_or_default(),
                            total_tokens: usage["total_tokens"].as_u64().unwrap_or_default(),
                        },
                    });
                }
                let diagnostic = response_diagnostic(&response);
                // Codex completion metadata does not require a status or output
                // array. The event type is the completion signal; an explicitly
                // contradictory status is still rejected.
                if event.kind != RESPONSE_COMPLETED
                    || response["status"]
                        .as_str()
                        .is_some_and(|status| status != "completed")
                {
                    return Err(AgentError::Model(format!(
                        "Responses {}: class={}, {diagnostic}",
                        event.kind,
                        classify_response_error(&response["error"])
                    )));
                }
                let mut normalized = response.clone();
                if !self.done_items.is_empty() {
                    for index in self.added_items.keys() {
                        if !self.done_items.contains_key(index) {
                            return Err(AgentError::Model(format!(
                                "Responses completed before output item {index} finished; {diagnostic}"
                            )));
                        }
                    }
                    // Codex semantics: completed items are authoritative. The
                    // terminal response is metadata, not another output snapshot.
                    normalized["output"] = json!(self.done_items.values().collect::<Vec<_>>());
                }
                // A completed-only provider can still supply its output here.
                self.completed = Some(
                    completed_response(&normalized)
                        .map_err(|error| AgentError::Model(format!("{error}; {diagnostic}")))?,
                );
            }
            ERROR => {
                return Err(AgentError::Model(format!(
                    "Responses stream error: code={}, message={}",
                    event.code.as_deref().unwrap_or("unknown"),
                    event.message.as_deref().unwrap_or("unknown")
                )))
            }
            _ => {} // Unknown extensions, reasoning deltas, and lifecycle metadata.
        }
        Ok(())
    }

    fn item_index(&self, explicit: Option<u64>, item: &Value) -> u64 {
        explicit
            .or_else(|| {
                item["id"].as_str().and_then(|id| {
                    self.added_items
                        .iter()
                        .chain(self.done_items.iter())
                        .find(|(_, existing)| existing["id"] == id)
                        .map(|(index, _)| *index)
                })
            })
            .unwrap_or_else(|| {
                self.added_items
                    .keys()
                    .chain(self.done_items.keys())
                    .max()
                    .map_or(0, |index| index.saturating_add(1))
            })
    }

    fn finish(self) -> Result<ModelResponse> {
        self.completed
            .ok_or_else(|| AgentError::Model("stream closed before response.completed".into()))
    }
}

fn check_item_identity(added: &Value, done: &Value) -> Result<()> {
    for key in ["type", "id", "call_id", "name"] {
        if let (Some(a), Some(b)) = (added.get(key), done.get(key)) {
            if a != b {
                return Err(AgentError::Model(format!(
                    "Responses item identity conflict: {key}"
                )));
            }
        }
    }
    Ok(())
}

// Codex distinguishes provider failures before retry policy is applied. Canary
// keeps the classification in diagnostics; it does not restart an emitted stream.
fn classify_response_error(error: &Value) -> &'static str {
    match error["code"].as_str() {
        Some("context_length_exceeded" | "context_window_exceeded") => "context_window",
        Some("insufficient_quota" | "quota_exceeded") => "quota",
        Some("rate_limit_exceeded") => "rate_limit",
        Some("server_overloaded" | "overloaded") => "overloaded",
        Some("invalid_request" | "invalid_request_error") => "invalid_request",
        _ => "stream",
    }
}

fn response_diagnostic(response: &Value) -> Value {
    json!({
        "response_id": response["id"],
        "status": response["status"],
        "incomplete_details": response["incomplete_details"],
        "error_code": response["error"]["code"],
        "usage": response["usage"],
        "max_output_tokens": response["max_output_tokens"],
        "output_types": response["output"].as_array().map(|items|
            items.iter().map(|item| item["type"].clone()).collect::<Vec<_>>()),
    })
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| AgentError::Model(format!("Responses item missing string {key}")))
}

fn completed_response(response: &Value) -> Result<ModelResponse> {
    let output = response["output"]
        .as_array()
        .ok_or_else(|| AgentError::Model("Responses completion missing output array".into()))?;
    let mut text = String::new();
    let mut calls = Vec::new();
    for item in output {
        if item["status"] == "incomplete" || item["status"] == "in_progress" {
            return Err(AgentError::Model(format!(
                "Responses output item is not complete: id={}",
                item["id"]
            )));
        }
        match item["type"].as_str() {
            Some("message") => {
                let parts = item["content"]
                    .as_array()
                    .ok_or_else(|| AgentError::Model("Responses message missing content".into()))?;
                for part in parts {
                    match part["type"].as_str() {
                        Some("output_text") => text.push_str(required_string(part, "text")?),
                        Some("refusal") => text.push_str(required_string(part, "refusal")?),
                        _ => {
                            return Err(AgentError::Model(
                                "unsupported Responses message content".into(),
                            ))
                        }
                    }
                }
            }
            Some("function_call") => {
                let arguments = required_string(item, "arguments")?;
                calls.push(ModelFunctionCall {
                    call_id: required_string(item, "call_id")?.to_owned(),
                    name: required_string(item, "name")?.to_owned(),
                    arguments: serde_json::from_str(arguments).map_err(|e| {
                        AgentError::Model(format!(
                            "invalid Responses function arguments: call_id={}, name={}, bytes={}, category={:?}, {e}",
                            item["call_id"], item["name"], arguments.len(), e.classify()
                        ))
                    })?,
                });
            }
            Some("reasoning") => {}
            _ => {
                return Err(AgentError::Model(
                    "unsupported Responses output item".into(),
                ))
            }
        }
    }
    if text.is_empty() && calls.is_empty() {
        return Err(AgentError::Model(
            "Responses completion has no text or function calls".into(),
        ));
    }
    Ok(ModelResponse::Assistant {
        text: (!text.is_empty()).then_some(text),
        function_calls: calls,
        continuation: Some(ModelContinuation {
            format: CONTINUATION_FORMAT.into(),
            payload: Value::Array(output.clone()),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ModelConfig {
        ModelConfig {
            base_url: "http://localhost/v1".into(),
            api_key: "test".into(),
            model: "test-model".into(),
            reasoning_effort: "high".into(),
        }
    }

    fn output() -> Value {
        json!([
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
            {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "Checking.", "annotations": []}]},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup",
             "arguments": "{\"q\":1}", "status": "completed"},
            {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "lookup",
             "arguments": "{}", "status": "completed"}
        ])
    }

    fn completed(output: Value) -> Value {
        json!({"type": "response.completed", "response": {
            "status": "completed", "output": output,
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                      "input_tokens_details": {"cached_tokens": 3}}
        }})
    }

    #[test]
    fn continuation_replays_ordered_items_without_duplicate_calls_or_text() {
        let ModelResponse::Assistant {
            text,
            function_calls,
            continuation,
        } = completed_response(&completed(output())["response"]).unwrap()
        else {
            panic!("assistant")
        };
        assert_eq!(function_calls.len(), 2);
        let body = request_body(
            &config(),
            ModelRequest {
                messages: vec![
                    ChatMessage::Assistant {
                        content: text,
                        tool_calls: function_calls,
                        continuation,
                    },
                    ChatMessage::Tool {
                        tool_call_id: "call_1".into(),
                        name: "lookup".into(),
                        content: json!("found"),
                    },
                    ChatMessage::Tool {
                        tool_call_id: "call_2".into(),
                        name: "lookup".into(),
                        content: json!({"ok": true}),
                    },
                ],
                functions: vec![],
            },
        )
        .unwrap();
        let input = body["input"].as_array().unwrap();
        assert_eq!(&input[..4], output().as_array().unwrap());
        assert_eq!(input.len(), 6);
        assert_eq!(input[4]["call_id"], "call_1");
        assert_eq!(input[4]["output"], "found");
        assert_eq!(body["store"], false);
        assert_eq!(body["reasoning"]["effort"], "high");
        assert!(body.get("tools").is_none());
        assert!(body.get("previous_response_id").is_none());
    }

    #[test]
    fn legacy_history_and_function_schema_are_supported() {
        let assistant: ChatMessage = serde_json::from_value(json!({
            "role": "assistant", "content": "checking",
            "tool_calls": [{"call_id": "c1", "name": "lookup", "arguments": {}}]
        }))
        .unwrap();
        let body = request_body(
            &config(),
            ModelRequest {
                messages: vec![assistant],
                functions: vec![canary_agent_kernel::FunctionSpec {
                    name: "lookup".into(),
                    description: "lookup".into(),
                    parameters: json!({"type": "object", "properties": {}}),
                }],
            },
        )
        .unwrap();
        assert_eq!(body["input"][1]["type"], "function_call");
        assert_eq!(body["input"][1]["call_id"], "c1");
        assert_eq!(body["tools"][0]["name"], "lookup");
        assert_eq!(body["tools"][0]["strict"], false);
        assert!(body["tools"][0].get("function").is_none());
    }

    #[test]
    fn rejects_foreign_or_malformed_continuation() {
        for continuation in [
            ModelContinuation {
                format: "other.v1".into(),
                payload: json!([]),
            },
            ModelContinuation {
                format: CONTINUATION_FORMAT.into(),
                payload: json!({}),
            },
        ] {
            assert!(request_body(
                &config(),
                ModelRequest {
                    messages: vec![ChatMessage::Assistant {
                        content: None,
                        tool_calls: vec![],
                        continuation: Some(continuation),
                    }],
                    functions: vec![],
                }
            )
            .is_err());
        }
    }

    #[tokio::test]
    async fn streaming_progress_usage_and_utf8_framing() {
        let events = [
            json!({"type": "response.created"}),
            json!({"type": "response.output_text.delta", "delta": ""}),
            json!({"type": "response.output_text.delta", "delta": "你好"}),
            json!({"type": "response.output_item.added", "item": {
                "type": "function_call", "call_id": "call_1", "name": "lookup"
            }}),
            json!({"type": "response.function_call_arguments.delta", "delta": "{}"}),
            json!({"type": "response.function_call_arguments.done", "arguments": "{}"}),
            completed(output()),
        ];
        let stream = events
            .iter()
            .map(|event| format!("event: ignored\r\ndata: {event}\r\n\r\n"))
            .collect::<String>();
        let mut emitted = Vec::new();
        // Split at every byte, including inside UTF-8 and SSE delimiters.
        let stream = futures_util::stream::iter(
            stream
                .bytes()
                .map(|byte| Ok::<_, std::io::Error>(vec![byte])),
        );
        read_response_stream(stream, Duration::from_secs(1), &mut |e| emitted.push(e))
            .await
            .unwrap();
        assert_eq!(
            emitted
                .iter()
                .filter(|e| matches!(e, ModelStreamEvent::OutputProgress))
                .count(),
            3
        );
        assert_eq!(
            emitted
                .iter()
                .filter(|e| matches!(e, ModelStreamEvent::AssistantDelta { .. }))
                .count(),
            1
        );
        assert!(
            matches!(emitted.last(), Some(ModelStreamEvent::TokenUsage { usage })
            if usage.output_tokens == 20 && usage.cached_input_tokens == 3)
        );
    }

    #[test]
    fn failed_incomplete_and_truncated_streams_do_not_succeed() {
        for kind in ["response.failed", "response.incomplete", "error"] {
            let mut decoder = ResponseDecoder::default();
            assert!(decoder
                .frame(&format!("data: {}", json!({"type": kind})), &mut |_| {})
                .is_err());
        }
        let mut decoder = ResponseDecoder::default();
        decoder
            .frame(
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}",
                &mut |_| {},
            )
            .unwrap();
        decoder.frame("data: [DONE]", &mut |_| {}).unwrap();
        assert!(decoder.finish().is_err());
        let invalid =
            json!([{"type": "function_call", "call_id": "c", "name": "lookup", "arguments": "{"}]);
        assert!(completed_response(&completed(invalid)["response"]).is_err());
    }

    fn feed(
        decoder: &mut ResponseDecoder,
        event: Value,
        emitted: &mut Vec<ModelStreamEvent>,
    ) -> Result<()> {
        decoder.frame(&format!("data: {event}"), &mut |e| emitted.push(e))
    }

    #[test]
    fn reconstructs_empty_terminal_output_from_completed_items_in_order() {
        let mut decoder = ResponseDecoder::default();
        let mut emitted = Vec::new();
        // Arrival order does not determine output order.
        for index in [3, 1, 0, 2] {
            feed(
                &mut decoder,
                json!({
                    "type": OUTPUT_ITEM_DONE, "output_index": index,
                    "item": output()[index],
                }),
                &mut emitted,
            )
            .unwrap();
        }
        feed(&mut decoder, completed(json!([])), &mut emitted).unwrap();
        let ModelResponse::Assistant {
            text,
            function_calls,
            continuation,
        } = decoder.finish().unwrap()
        else {
            panic!("assistant")
        };
        assert_eq!(text.as_deref(), Some("Checking."));
        assert_eq!(function_calls.len(), 2);
        assert_eq!(continuation.unwrap().payload, output());
        // Snapshots must not double-count progress or replay streamed text.
        assert_eq!(emitted.len(), 1);
        assert!(matches!(emitted[0], ModelStreamEvent::TokenUsage { .. }));
    }

    #[test]
    fn recovers_truncated_terminal_arguments_from_arguments_done() {
        let mut decoder = ResponseDecoder::default();
        let mut emitted = Vec::new();
        let mut call = output()[2].clone();
        call["arguments"] = json!("");
        call["status"] = json!("in_progress");
        feed(
            &mut decoder,
            json!({
                "type": OUTPUT_ITEM_ADDED, "output_index": 0, "item": call
            }),
            &mut emitted,
        )
        .unwrap();
        feed(
            &mut decoder,
            json!({
                "type": FUNCTION_ARGUMENTS_DONE, "output_index": 0,
                "item_id": "fc_1", "arguments": "{\"q\":1}"
            }),
            &mut emitted,
        )
        .unwrap();
        let mut truncated = output()[2].clone();
        truncated["arguments"] = json!("{\"q\":");
        feed(&mut decoder, completed(json!([truncated])), &mut emitted).unwrap();
        let ModelResponse::Assistant {
            function_calls,
            continuation,
            ..
        } = decoder.finish().unwrap()
        else {
            panic!("assistant")
        };
        assert_eq!(function_calls[0].arguments, json!({"q": 1}));
        assert_eq!(continuation.unwrap().payload[0]["arguments"], "{\"q\":1}");
    }

    #[test]
    fn empty_reasoning_only_and_partial_json_report_metadata_and_preserve_usage() {
        for items in [
            json!([]),
            json!([output()[0].clone()]),
            json!([{"type": "function_call", "call_id": "c", "name": "lookup", "arguments": "{\"q\":"}]),
        ] {
            let mut decoder = ResponseDecoder::default();
            let mut emitted = Vec::new();
            let mut event = completed(items);
            event["response"]["id"] = json!("resp_diagnostic");
            let error = feed(&mut decoder, event, &mut emitted)
                .unwrap_err()
                .to_string();
            assert!(error.contains("resp_diagnostic"));
            assert!(error.contains("output_types"));
            assert!(error.contains("usage"));
            assert!(!error.contains("opaque")); // No reasoning payload in diagnostics.
            assert_eq!(emitted.len(), 1);
            assert!(matches!(emitted[0], ModelStreamEvent::TokenUsage { .. }));
        }
    }

    #[test]
    fn incomplete_responses_fail_and_completed_items_override_terminal_snapshots() {
        let mut decoder = ResponseDecoder::default();
        let mut emitted = Vec::new();
        feed(
            &mut decoder,
            json!({
                "type": OUTPUT_ITEM_DONE, "output_index": 0, "item": output()[2]
            }),
            &mut emitted,
        )
        .unwrap();
        let mut incomplete = completed(json!([]));
        incomplete["type"] = json!(RESPONSE_INCOMPLETE);
        incomplete["response"]["status"] = json!("incomplete");
        incomplete["response"]["incomplete_details"] = json!({"reason": "max_output_tokens"});
        let error = feed(&mut decoder, incomplete, &mut emitted)
            .unwrap_err()
            .to_string();
        assert!(error.contains("max_output_tokens"));
        assert!(decoder.completed.is_none());

        for change in [
            json!({"call_id": "other"}),
            json!({"arguments": "{\"q\":2}"}),
            json!({"status": "incomplete"}),
        ] {
            let mut terminal = output()[2].clone();
            for (key, value) in change.as_object().unwrap() {
                terminal[key] = value.clone();
            }
            feed(&mut decoder, completed(json!([terminal])), &mut emitted).unwrap();
            assert!(
                matches!(&decoder.completed, Some(ModelResponse::Assistant { function_calls, .. })
                if function_calls[0].call_id == "call_1" && function_calls[0].arguments == json!({"q": 1}))
            );
        }
    }

    // Adapted from Codex SSE tests: completed items are independent of terminal
    // metadata; response.completed need not contain an output array or status.
    #[tokio::test]
    async fn completed_items_with_metadata_only_completion() {
        let mut frames = String::from(": heartbeat\n\n");
        for item in output().as_array().unwrap() {
            frames.push_str(&format!(
                "data: {}\n\n",
                json!({"type": OUTPUT_ITEM_DONE, "item": item})
            ));
        }
        frames.push_str(&format!(
            "data: {}\n\n",
            json!({
                "type": RESPONSE_COMPLETED, "response": {"id": "response_1"}
            })
        ));
        let stream = futures_util::stream::iter([Ok::<_, std::io::Error>(frames.into_bytes())]);
        let result = read_response_stream(stream, Duration::from_secs(1), &mut |_| {})
            .await
            .unwrap();
        assert!(matches!(result, ModelResponse::Assistant {
            continuation: Some(ModelContinuation { payload, .. }), ..
        } if payload == output()));
    }

    #[tokio::test]
    async fn stream_idle_timeout_and_early_close() {
        let pending =
            futures_util::stream::pending::<std::result::Result<Vec<u8>, std::io::Error>>();
        let error = read_response_stream(pending, Duration::from_millis(10), &mut |_| {})
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("idle timeout"));

        for payload in [
            "data: [DONE]\n\n".to_owned(),
            format!(
                "data: {}\n\n",
                json!({"type": OUTPUT_ITEM_DONE, "item": output()[1]})
            ),
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\"}}".to_owned(),
        ] {
            let stream =
                futures_util::stream::iter([Ok::<_, std::io::Error>(payload.into_bytes())]);
            assert!(
                read_response_stream(stream, Duration::from_secs(1), &mut |_| {})
                    .await
                    .is_err()
            );
        }
    }

    #[tokio::test]
    async fn malformed_and_transport_errors_are_not_silently_ignored() {
        let malformed =
            futures_util::stream::iter([Ok::<_, std::io::Error>(b"data: {\n\n".to_vec())]);
        assert!(
            read_response_stream(malformed, Duration::from_secs(1), &mut |_| {})
                .await
                .is_err()
        );
        let broken = futures_util::stream::iter([Err::<Vec<u8>, _>(std::io::Error::other(
            "broken connection",
        ))]);
        assert!(
            read_response_stream(broken, Duration::from_secs(1), &mut |_| {})
                .await
                .unwrap_err()
                .to_string()
                .contains("broken connection")
        );
    }

    #[test]
    fn unfinished_items_and_identity_mismatch_fail() {
        let mut decoder = ResponseDecoder::default();
        let mut events = Vec::new();
        feed(
            &mut decoder,
            json!({"type": OUTPUT_ITEM_ADDED, "output_index": 0, "item": output()[2]}),
            &mut events,
        )
        .unwrap();
        let mut mismatch = output()[2].clone();
        mismatch["call_id"] = json!("wrong");
        assert!(feed(
            &mut decoder,
            json!({"type": OUTPUT_ITEM_DONE, "output_index": 0, "item": mismatch}),
            &mut events
        )
        .is_err());
        feed(
            &mut decoder,
            json!({"type": OUTPUT_ITEM_DONE, "output_index": 1, "item": output()[1]}),
            &mut events,
        )
        .unwrap();
        assert!(feed(&mut decoder, completed(json!([])), &mut events).is_err());
    }
    #[tokio::test]
    async fn client_posts_responses_request_and_decodes_stream() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            let header_end = loop {
                let count = socket.read(&mut chunk).unwrap();
                assert!(count > 0);
                request.extend_from_slice(&chunk[..count]);
                if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    break end + 4;
                }
            };
            let headers = String::from_utf8(request[..header_end].to_vec()).unwrap();
            assert!(headers.starts_with("POST /v1/responses HTTP/1.1"));
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|s| s.trim().parse().unwrap())
                })
                .unwrap();
            while request.len() < header_end + length {
                let count = socket.read(&mut chunk).unwrap();
                assert!(count > 0);
                request.extend_from_slice(&chunk[..count]);
            }
            let body: Value =
                serde_json::from_slice(&request[header_end..header_end + length]).unwrap();
            assert_eq!(body["store"], false);
            assert_eq!(body["reasoning"]["effort"], "high");
            let response = format!("data: {}\n\n", completed(output()));
            write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", response.len(), response).unwrap();
        });
        let mut config = config();
        config.base_url = format!("http://{address}/v1");
        let client = ResponsesClient::new(config);
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client.stream_complete(
                ModelRequest {
                    messages: vec![],
                    functions: vec![],
                },
                &mut |_| {},
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(matches!(
            result,
            ModelResponse::Assistant {
                continuation: Some(_),
                ..
            }
        ));
        server.join().unwrap();
    }
}
