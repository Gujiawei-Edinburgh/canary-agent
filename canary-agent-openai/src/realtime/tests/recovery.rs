use super::*;

struct Replay {
    events: Vec<ModelSessionEvent>,
    error: String,
}

async fn replay(events: Vec<Value>) -> Replay {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        socket.next().await.unwrap().unwrap();
        for event in std::iter::once(json!({"type":"session.updated"}))
            .chain(events)
            .chain(std::iter::once(
                json!({"type":"error","error":{"message":"end-of-test"}}),
            ))
        {
            socket
                .send(Message::Text(event.to_string().into()))
                .await
                .unwrap();
        }
        assert!(socket.next().await.unwrap().unwrap().is_close());
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
    let mut events = Vec::new();
    let error = loop {
        match session.receiver.receive().await {
            Ok(Some(event)) => events.push(event),
            Err(error) => break error.to_string(),
            Ok(None) => panic!("unexpected EOF"),
        }
    };
    session
        .sender
        .send(ModelSessionCommand::Close)
        .await
        .unwrap();
    server.await.unwrap();
    Replay { events, error }
}

fn truncated_item() -> Value {
    json!({"id":"item","type":"function_call","status":"completed","call_id":"call","name":"exec_command","arguments":"{\"cmd\":\"echo"})
}

#[tokio::test]
async fn reconstructs_arguments_without_executing_partial_snapshots() {
    for include_done in [false, true] {
        let mut events = vec![
            json!({"type":"response.created","response":{"id":"response"}}),
            json!({"type":"response.function_call_arguments.delta","response_id":"response","item_id":"item","delta":"{\"cmd\":\"echo "}),
            json!({"type":"response.function_call_arguments.delta","response_id":"response","item_id":"item","delta":"hello\"}"}),
        ];
        if include_done {
            events.push(json!({"type":"response.function_call_arguments.done","response_id":"response","item_id":"item","arguments":"{\"cmd\":\"echo authoritative\"}"}));
        }
        events.extend([
            json!({"type":"response.output_item.done","response_id":"response","item":truncated_item()}),
            json!({"type":"response.done","response":{"id":"response","status":"completed","output":[truncated_item()]}}),
        ]);
        let result = replay(events).await;
        assert!(result.error.contains("end-of-test"), "{}", result.error);
        let calls: Vec<_> = result
            .events
            .iter()
            .filter_map(|e| match e {
                ModelSessionEvent::ItemCompleted { function_calls, .. } => Some(function_calls),
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].arguments["cmd"],
            if include_done {
                "echo authoritative"
            } else {
                "echo hello"
            }
        );
    }
}

#[tokio::test]
async fn cancelled_and_failed_responses_do_not_dispatch_truncated_calls() {
    for status in ["cancelled", "incomplete", "failed"] {
        let result = replay(vec![
            json!({"type":"response.created","response":{"id":"response"}}),
            json!({"type":"response.output_item.done","response_id":"response","item":truncated_item()}),
            json!({"type":"response.done","response":{"id":"response","status":status,"output":[truncated_item()],"usage":{"input_tokens":10,"output_tokens":2}}}),
        ]).await;
        assert!(result.error.contains("end-of-test"), "{}", result.error);
        assert!(!result
            .events
            .iter()
            .any(|e| matches!(e, ModelSessionEvent::ItemCompleted { .. })));
        assert!(result
            .events
            .iter()
            .any(|e| matches!(e, ModelSessionEvent::Usage { .. })));
        assert!(result
            .events
            .iter()
            .any(|e| matches!(e, ModelSessionEvent::ResponseFinished { .. })));
    }
}

#[tokio::test]
async fn malformed_completed_call_fails_with_identifiers_not_generic_json_eof() {
    let result = replay(vec![
        json!({"type":"response.created","response":{"id":"response"}}),
        json!({"type":"response.output_item.done","response_id":"response","item":truncated_item()}),
        json!({"type":"response.done","response":{"id":"response","status":"completed","output":[truncated_item()]}}),
    ]).await;
    assert!(result
        .error
        .contains("invalid completed realtime function arguments"));
    assert!(result.error.contains("call=call"));
    assert!(!result
        .events
        .iter()
        .any(|e| matches!(e, ModelSessionEvent::ItemCompleted { .. })));
}
