use super::*;
use canary_agent_kernel::{
    FunctionSpec, ModelRequest, Thread, ToolResult, TurnItem, TurnItemKind, TurnItemSource,
};
use canary_agent_runtime::{ModelInput, ModelResponseStatus};

mod recovery;

fn config() -> RealtimeConfig {
    RealtimeConfig::new("https://example.com/compatible-mode/v1", "secret", "model")
}

#[test]
fn normalizes_endpoint_and_redacts_credentials() {
    let config = config();
    assert_eq!(
        config.websocket_url().unwrap().as_str(),
        "wss://example.com/v1/realtime?model=model"
    );
    assert!(!format!("{config:?}").contains("secret"));
}

#[test]
fn openai_session_uses_ga_schema() {
    let request = ModelRequest {
        messages: vec![],
        functions: vec![FunctionSpec {
            name: "test".into(),
            description: "test".into(),
            parameters: json!({"type":"object"}),
        }],
    };
    let openai = protocol::session_update(&config(), &request);
    assert_eq!(openai["session"]["tools"][0]["name"], "test");
    assert_eq!(openai["session"]["audio"]["input"]["format"]["rate"], 24000);
    assert_eq!(
        openai["session"]["audio"]["input"]["turn_detection"]["interrupt_response"],
        true
    );
}

#[test]
fn acceptance_does_not_answer_native_call_and_result_requests_continuation() {
    let mut state = protocol::State::new(Default::default(), false);
    let config = config();
    let accepted = TurnItem::new(
        TurnItemSource::Runtime,
        TurnItemKind::ToolAccepted {
            call_id: "call".into(),
            name: "test".into(),
            operation_id: "op".into(),
        },
    );
    assert!(protocol::command(
        &config,
        &mut state,
        ModelSessionCommand::Observe {
            items: vec![accepted]
        }
    )
    .unwrap()
    .is_empty());
    let result = TurnItem::new(
        TurnItemSource::Tool,
        TurnItemKind::ToolCompleted {
            call_id: "call".into(),
            name: "test".into(),
            operation_id: "op".into(),
            result: ToolResult::Success {
                output: json!({"ok":true}),
            },
        },
    );
    let frames = protocol::command(
        &config,
        &mut state,
        ModelSessionCommand::Observe {
            items: vec![result.clone()],
        },
    )
    .unwrap();
    assert_eq!(frames[0]["item"]["type"], "function_call_output");
    assert_eq!(frames[1]["type"], "response.create");
    assert!(protocol::command(
        &config,
        &mut state,
        ModelSessionCommand::Observe {
            items: vec![result]
        }
    )
    .unwrap()
    .is_empty());
    assert!(protocol::check_error(&json!({"type":"error","error":{"message":"failed"}})).is_err());
    assert!(protocol::command(
        &config,
        &mut state,
        ModelSessionCommand::Input(ModelInput::Audio {
            chunk: canary_agent_runtime::AudioChunk {
                format: {
                    let mut format = config.input_format();
                    format.sample_rate = 16000;
                    format
                },
                data: vec![0, 0]
            }
        })
    )
    .is_err());
}

#[test]
fn ordinary_closed_output_does_not_end_the_session_or_duplicate_response_creation() {
    let config = config();
    let result = TurnItem::new(
        TurnItemSource::Tool,
        TurnItemKind::ToolCompleted {
            call_id: "file-close".into(),
            name: "close_file".into(),
            operation_id: "op".into(),
            result: ToolResult::Success {
                output: json!({"status":"closed"}),
            },
        },
    );
    let mut state = protocol::State::new(Default::default(), false);
    let frames = protocol::command(
        &config,
        &mut state,
        ModelSessionCommand::Observe {
            items: vec![result],
        },
    )
    .unwrap();
    assert_eq!(frames.last().unwrap()["type"], "response.create");
    let frames = protocol::command(
        &config,
        &mut state,
        ModelSessionCommand::Input(ModelInput::Text {
            text: "also this".into(),
        }),
    )
    .unwrap();
    assert!(!frames
        .iter()
        .any(|frame| frame["type"] == "response.create"));
}

#[test]
fn restores_pending_native_calls_without_fabricating_success() {
    use canary_agent_kernel::ChatMessage;
    let request = ModelRequest {
        messages: vec![
            ChatMessage::Assistant {
                continuation: None,
                content: None,
                tool_calls: vec![canary_agent_kernel::ModelFunctionCall {
                    call_id: "pending".into(),
                    name: "test".into(),
                    arguments: json!({}),
                }],
            },
            ChatMessage::Tool {
                tool_call_id: "pending".into(),
                name: "test".into(),
                content: json!({"status":"accepted","operation_id":"op"}),
            },
        ],
        functions: vec![],
    };
    let config = ModelSessionConfig {
        connection_id: "conn".into(),
        turn_id: "turn".into(),
        request,
        history: Thread::new("thread"),
        model: ModelDescriptor {
            fqn: "test".into(),
            settings: json!({}),
        },
    };
    let restored = protocol::restore(&config).unwrap();
    assert!(restored.waiting_calls.contains("pending"));
    assert_eq!(restored.frames.len(), 1);
    assert_eq!(restored.frames[0]["item"]["type"], "function_call");
}

#[tokio::test]
async fn websocket_tool_result_precedes_its_continuation() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        socket.next().await.unwrap().unwrap();
        for event in [
            json!({"type":"session.updated"}),
            json!({"type":"response.created","response":{"id":"r1"}}),
            json!({"type":"response.output_item.done","response_id":"r1","item":{"id":"item1","type":"function_call","call_id":"c1","name":"test","arguments":"{}"}}),
            json!({"type":"response.done","response":{"id":"r1","status":"completed","output":[]}}),
        ] {
            socket
                .send(Message::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        let output: Value =
            serde_json::from_str(socket.next().await.unwrap().unwrap().to_text().unwrap()).unwrap();
        assert_eq!(output["item"]["type"], "function_call_output");
        assert_eq!(output["item"]["call_id"], "c1");
        let continuation: Value =
            serde_json::from_str(socket.next().await.unwrap().unwrap().to_text().unwrap()).unwrap();
        assert_eq!(continuation["type"], "response.create");
        socket
            .send(Message::Text(
                json!({"type":"response.created","response":{"id":"r2"}})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        while let Some(Ok(message)) = socket.next().await {
            if message.is_close() {
                break;
            }
        }
    });
    let client = RealtimeClient::new(RealtimeConfig::new(
        format!("ws://{address}/realtime"),
        "secret",
        "model",
    ));
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
    assert!(matches!(
        session.receiver.receive().await.unwrap(),
        Some(ModelSessionEvent::ResponseStarted { .. })
    ));
    assert!(
        matches!(session.receiver.receive().await.unwrap(),Some(ModelSessionEvent::ItemCompleted { function_calls,.. }) if function_calls[0].call_id == "c1")
    );
    assert!(matches!(
        session.receiver.receive().await.unwrap(),
        Some(ModelSessionEvent::ResponseFinished { .. })
    ));
    let item = TurnItem::new(
        TurnItemSource::Tool,
        TurnItemKind::ToolCompleted {
            call_id: "c1".into(),
            name: "test".into(),
            operation_id: "op".into(),
            result: ToolResult::Success {
                output: json!({"ok":true}),
            },
        },
    );
    session
        .sender
        .send(ModelSessionCommand::Observe { items: vec![item] })
        .await
        .unwrap();
    assert!(
        matches!(session.receiver.receive().await.unwrap(),Some(ModelSessionEvent::ResponseStarted { response_id }) if response_id == "r2")
    );
    session
        .sender
        .send(ModelSessionCommand::Close)
        .await
        .unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn websocket_adapter_exchanges_audio_tools_and_interruption() {
    use tokio::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        let update = socket.next().await.unwrap().unwrap();
        let value: Value = serde_json::from_str(update.to_text().unwrap()).unwrap();
        assert_eq!(value["type"], "session.update");
        socket
            .send(Message::Text(
                json!({"type":"session.updated"}).to_string().into(),
            ))
            .await
            .unwrap();
        let input = socket.next().await.unwrap().unwrap();
        let value: Value = serde_json::from_str(input.to_text().unwrap()).unwrap();
        assert_eq!(value["type"], "input_audio_buffer.append");
        for event in [
            json!({"type":"response.created","response":{"id":"r1"}}),
            json!({"type":"response.audio.delta","response_id":"r1","item_id":"a","delta":"AAA="}),
            json!({"type":"input_audio_buffer.speech_started","item_id":"u1"}),
            json!({"type":"conversation.item.input_audio_transcription.completed","item_id":"u1","transcript":""}),
            json!({"type":"response.output_item.done","response_id":"r1","item":{"id":"late","type":"function_call","name":"unsafe","call_id":"c","arguments":"{}"}}),
            json!({"type":"response.done","response":{"id":"r1","status":"cancelled","output":[],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}),
        ] {
            socket
                .send(Message::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        // Keep the writer alive while the client consumes queued events.
        while let Some(Ok(message)) = socket.next().await {
            if message.is_close() {
                break;
            }
        }
    });
    let client = RealtimeClient::new(RealtimeConfig::new(
        format!("ws://{address}/realtime"),
        "secret",
        "model",
    ));
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
            chunk: canary_agent_runtime::AudioChunk {
                format: client.config.input_format(),
                data: vec![0, 0],
            },
        }))
        .await
        .unwrap();
    let mut events = Vec::new();
    for _ in 0..6 {
        events.push(session.receiver.receive().await.unwrap().unwrap());
    }
    assert!(events.iter().any(|e| matches!(
        e,
        ModelSessionEvent::ResponseFinished {
            status: ModelResponseStatus::Interrupted { .. },
            ..
        }
    )));
    assert!(events
        .iter()
        .any(|e| matches!(e, ModelSessionEvent::Usage { usage, .. } if usage.total_tokens == 5)));
    assert!(!events
        .iter()
        .any(|e| matches!(e, ModelSessionEvent::ItemCompleted { .. })));
    assert!(events.iter().any(
        |e| matches!(e,ModelSessionEvent::TranscriptFinalized { text,.. } if text.is_empty())
    ));
    session
        .sender
        .send(ModelSessionCommand::Close)
        .await
        .unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn reset_during_setup_reports_connection_phase() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        socket.next().await.unwrap().unwrap();
        // Drop TCP without a WebSocket Close frame, as in the reported reset.
    });
    let client = RealtimeClient::new(RealtimeConfig::new(
        format!("ws://{address}/realtime"),
        "secret",
        "model",
    ));
    let result = client
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
        .await;
    let error = result
        .err()
        .expect("reset must not be accepted")
        .to_string();
    assert!(error.contains("waiting for session.updated"), "{error}");
    server.await.unwrap();
}
