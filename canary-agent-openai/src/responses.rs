use crate::config::{ModelConfig, RetryConfig};
use crate::transport::{find_sse_frame_end, send_with_retries};
use canary_agent_kernel::{
    ChatMessage, ModelContinuation, ModelFunctionCall, ModelRequest, ModelResponse,
    ModelStreamEvent, TokenUsage,
};
use canary_agent_runtime::model::{ModelClient, ModelDescriptor, ModelStreamHandler};
use canary_agent_runtime::{AgentError, Result};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;

const CONTINUATION_FORMAT: &str = "openai.responses.v1";

/// Stateless Responses API client. Continuation items travel through model history,
/// rather than an in-memory cache or a server-side conversation.
#[derive(Debug, Clone)]
pub struct ResponsesClient {
    http: reqwest::Client,
    config: ModelConfig,
    retry: RetryConfig,
}

impl ResponsesClient {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            config,
            retry: RetryConfig::default(),
        }
    }

    pub fn with_retry_config(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
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
            let mut stream = response.bytes_stream();
            let mut buffer = Vec::new();
            let mut decoder = ResponseDecoder::default();
            while let Some(chunk) = stream.next().await {
                buffer.extend_from_slice(&chunk.map_err(|e| AgentError::Http(e.to_string()))?);
                decoder.consume(&mut buffer, on_event)?;
                if let Some(response) = decoder.completed.take() {
                    return Ok(response);
                }
            }
            // EOF is not success: a terminal response.completed event is required.
            if !buffer.iter().all(u8::is_ascii_whitespace) {
                let frame = std::str::from_utf8(&buffer)
                    .map_err(|e| AgentError::Model(format!("invalid Responses SSE UTF-8: {e}")))?;
                decoder.frame(frame, on_event)?;
            }
            decoder.finish()
        })
    }
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
}

impl ResponseDecoder {
    fn consume(&mut self, buffer: &mut Vec<u8>, emit: &mut ModelStreamHandler<'_>) -> Result<()> {
        while let Some((end, delimiter)) = find_sse_frame_end(buffer) {
            let frame = std::str::from_utf8(&buffer[..end])
                .map_err(|e| AgentError::Model(format!("invalid Responses SSE UTF-8: {e}")))?;
            self.frame(frame, emit)?;
            buffer.drain(..end + delimiter);
            if self.completed.is_some() {
                break;
            }
        }
        Ok(())
    }

    fn frame(&mut self, frame: &str, emit: &mut ModelStreamHandler<'_>) -> Result<()> {
        let data = frame
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(|line| line.strip_prefix(' ').unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n");
        if data.trim().is_empty() || data.trim() == "[DONE]" {
            return Ok(());
        }
        let event: Value = serde_json::from_str(&data)
            .map_err(|e| AgentError::Model(format!("invalid Responses SSE JSON: {e}")))?;
        match event["type"].as_str() {
            Some("response.output_text.delta" | "response.refusal.delta") => {
                let delta = required_string(&event, "delta")?;
                if !delta.is_empty() {
                    emit(ModelStreamEvent::OutputProgress);
                    emit(ModelStreamEvent::AssistantDelta {
                        text: delta.to_owned(),
                    });
                }
            }
            Some("response.function_call_arguments.delta") => {
                if !required_string(&event, "delta")?.is_empty() {
                    emit(ModelStreamEvent::OutputProgress);
                }
            }
            Some("response.output_item.added") => {
                let item = &event["item"];
                if item["type"] == "function_call"
                    && (item["name"].as_str().is_some_and(|s| !s.is_empty())
                        || item["call_id"].as_str().is_some_and(|s| !s.is_empty()))
                {
                    emit(ModelStreamEvent::OutputProgress);
                }
            }
            Some("response.completed") => {
                let response = &event["response"];
                if response["status"] != "completed" {
                    return Err(AgentError::Model(
                        "Responses terminal event has non-completed status".into(),
                    ));
                }
                let model_response = completed_response(response)?;
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
                self.completed = Some(model_response);
            }
            Some("response.failed" | "response.incomplete" | "error") => {
                return Err(AgentError::Model(format!(
                    "Responses stream failed: {event}"
                )));
            }
            Some(_) => {} // Lifecycle, reasoning, and duplicate done snapshots.
            None => return Err(AgentError::Model("Responses event missing type".into())),
        }
        Ok(())
    }

    fn finish(self) -> Result<ModelResponse> {
        self.completed.ok_or_else(|| {
            AgentError::Model("Responses stream ended before response.completed".into())
        })
    }
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
                        AgentError::Model(format!("invalid Responses function arguments: {e}"))
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

    #[test]
    fn streaming_progress_usage_and_utf8_framing() {
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
        let mut decoder = ResponseDecoder::default();
        let mut buffer = Vec::new();
        let mut emitted = Vec::new();
        // Split at every byte, including inside UTF-8 and SSE delimiters.
        for byte in stream.bytes() {
            buffer.push(byte);
            decoder
                .consume(&mut buffer, &mut |e| emitted.push(e))
                .unwrap();
        }
        assert!(decoder.finish().is_ok());
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
