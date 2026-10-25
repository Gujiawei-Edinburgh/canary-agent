mod common;
use canary_agent_kernel::{
    ChatMessage, ThreadProjection, TokenUsage, ToolResult, TurnItem, TurnItemKind, TurnStatus,
};
use canary_agent_runtime::*;
use common::*;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

struct OpenConnection {
    config: ModelSessionConfig,
    events: mpsc::Sender<ModelSessionEvent>,
    commands: mpsc::Receiver<ModelSessionCommand>,
    allow: oneshot::Sender<()>,
}

struct LiveModel(mpsc::UnboundedSender<OpenConnection>);
struct Writer(mpsc::Sender<ModelSessionCommand>);
struct Reader(mpsc::Receiver<ModelSessionEvent>);

impl ModelSessionSender for Writer {
    fn send(&mut self, command: ModelSessionCommand) -> ModelSessionFuture<'_, ()> {
        Box::pin(async move {
            self.0
                .send(command)
                .await
                .map_err(|_| AgentError::Model("test transport closed".into()))
        })
    }
}

impl ModelSessionReceiver for Reader {
    fn receive(&mut self) -> ModelSessionFuture<'_, Option<ModelSessionEvent>> {
        Box::pin(async move { Ok(self.0.recv().await) })
    }
}

impl ModelClient for LiveModel {
    fn model_descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            fqn: "test/live".into(),
            settings: json!({}),
        }
    }
    fn interaction(&self) -> ModelInteraction {
        ModelInteraction::Bidirectional
    }
    fn connect(&self, config: ModelSessionConfig) -> ModelSessionFuture<'_, ModelSession> {
        Box::pin(async move {
            let (events, incoming) = mpsc::channel(32);
            let (outgoing, commands) = mpsc::channel(32);
            let (allow, ready) = oneshot::channel();
            self.0
                .send(OpenConnection {
                    config,
                    events,
                    commands,
                    allow,
                })
                .unwrap_or_else(|_| panic!("test connection receiver"));
            ready
                .await
                .map_err(|_| AgentError::Model("test connection rejected".into()))?;
            Ok(ModelSession {
                sender: Box::new(Writer(outgoing)),
                receiver: Box::new(Reader(incoming)),
            })
        })
    }
}

struct OpenedInteraction {
    turn: RunningTurn,
    connection: OpenConnection,
}

struct LiveHarness {
    h: Harness,
    connections: mpsc::UnboundedReceiver<OpenConnection>,
}

impl LiveHarness {
    fn new() -> Self {
        Self::with_config(AgentConfig::default())
    }
    fn with_config(config: AgentConfig) -> Self {
        let (sender, connections) = mpsc::unbounded_channel();
        let h = Harness::with_model(config, Some(Arc::new(LiveModel(sender))));
        Self { h, connections }
    }
    async fn pending_connection(&mut self) -> OpenConnection {
        tokio::time::timeout(Duration::from_secs(2), self.connections.recv())
            .await
            .unwrap()
            .unwrap()
    }
    async fn open(&mut self) -> OpenedInteraction {
        let mut turn = self
            .h
            .agent
            .open_stream_turn("thread", json!({"owner":"audio"}), config())
            .await
            .unwrap();
        let mut connection = self.pending_connection().await;
        let (dummy, _) = oneshot::channel();
        std::mem::replace(&mut connection.allow, dummy)
            .send(())
            .unwrap();
        event(&mut turn, |e| matches!(e, TurnEvent::SessionReady { .. })).await;
        OpenedInteraction { turn, connection }
    }
}

impl OpenConnection {
    async fn emit(&self, event: ModelSessionEvent) {
        self.events.send(event).await.unwrap();
    }
    async fn command(&mut self) -> ModelSessionCommand {
        tokio::time::timeout(Duration::from_secs(2), self.commands.recv())
            .await
            .unwrap()
            .unwrap()
    }
    async fn observe(&mut self, predicate: impl Fn(&TurnItemKind) -> bool) -> Vec<TurnItem> {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let ModelSessionCommand::Observe { items } = self.command().await {
                    if items.iter().any(|item| predicate(&item.kind)) {
                        return items;
                    }
                }
            }
        })
        .await
        .unwrap()
    }
    async fn start(&self, id: &str) {
        self.emit(ModelSessionEvent::ResponseStarted {
            response_id: id.into(),
        })
        .await;
    }
    async fn item(&self, response: &str, item: &str, calls: Vec<ModelFunctionCall>) {
        self.emit(ModelSessionEvent::ItemCompleted {
            response_id: response.into(),
            item_id: item.into(),
            text: None,
            function_calls: calls,
        })
        .await;
    }
    async fn end_response(&self, id: &str) {
        self.emit(ModelSessionEvent::ResponseFinished {
            response_id: id.into(),
            status: ModelResponseStatus::Completed,
        })
        .await;
    }
    async fn close_turn(&self) {
        self.start("final-response").await;
        self.item(
            "final-response",
            "final-item",
            vec![call("close", "end_turn")],
        )
        .await;
        self.end_response("final-response").await;
    }
}

fn audio() -> AudioChunk {
    AudioChunk {
        format: AudioFormat {
            encoding: AudioEncoding::PcmS16Le,
            sample_rate: 16000,
            channels: 1,
        },
        data: vec![0, 1, 2, 3],
    }
}

#[tokio::test]
async fn text_ingress_is_durable_and_does_not_cancel_live_response() {
    let mut h = LiveHarness::new();
    let OpenedInteraction {
        mut turn,
        mut connection,
    } = h.open().await;
    connection.start("speaking").await;
    event(&mut turn, |e| {
        matches!(
            e,
            TurnEvent::Session {
                event: ModelSessionEvent::ResponseStarted { .. },
                ..
            }
        )
    })
    .await;
    turn.handle()
        .send_input(ModelInput::Text {
            text: "door opened".into(),
        })
        .await
        .unwrap();
    assert_eq!(
        connection.command().await,
        ModelSessionCommand::Input(ModelInput::Text {
            text: "door opened".into()
        })
    );
    assert_eq!(
        h.h.items()
            .iter()
            .filter(|item| matches!(
                &item.kind, TurnItemKind::UserInput { text, .. } if text == "door opened"
            ))
            .count(),
        1
    );
    assert!(connection.commands.try_recv().is_err());
    connection.end_response("speaking").await;
    connection.close_turn().await;
    assert_eq!(finish(turn).await, TurnOutcome::Completed);
}

#[tokio::test]
async fn text_delivery_failure_suspends_and_preserves_input() {
    let mut h = LiveHarness::new();
    let OpenedInteraction {
        mut turn,
        mut connection,
    } = h.open().await;
    connection.commands.close();
    // Local admission may succeed before the transport reports its failure.
    let _ = turn.handle().append_input("durable observation").await;
    event(&mut turn, |e| matches!(e, TurnEvent::Suspended { .. })).await;
    assert!(h.h.items().iter().any(|item| matches!(
        &item.kind, TurnItemKind::UserInput { text, .. } if text == "durable observation"
    )));
    assert!(turn.handle().append_input("not admitted").await.is_err());
    assert!(!h.h.items().iter().any(|item| matches!(
        &item.kind, TurnItemKind::UserInput { text, .. } if text == "not admitted"
    )));
    turn.handle().abort();
    assert!(matches!(finish(turn).await, TurnOutcome::Aborted { .. }));
}

#[tokio::test]
async fn continuous_input_and_transcripts_share_one_connection_without_reinference() {
    let mut h = LiveHarness::new();
    let OpenedInteraction {
        mut turn,
        mut connection,
    } = h.open().await;
    for _ in 0..2 {
        turn.handle()
            .send_input(ModelInput::Audio { chunk: audio() })
            .await
            .unwrap();
        assert_eq!(
            connection.command().await,
            ModelSessionCommand::Input(ModelInput::Audio { chunk: audio() })
        );
    }
    connection
        .emit(ModelSessionEvent::InputStarted {
            item_id: "utterance".into(),
        })
        .await;
    connection
        .emit(ModelSessionEvent::InputEnded {
            item_id: "utterance".into(),
        })
        .await;
    connection
        .emit(ModelSessionEvent::InputCommitted {
            item_id: "utterance".into(),
        })
        .await;
    for _ in 0..2 {
        connection
            .emit(ModelSessionEvent::TranscriptFinalized {
                item_id: "utterance".into(),
                text: "hello".into(),
            })
            .await;
    }
    event(&mut turn, |e| matches!(e, TurnEvent::ItemCommitted { item } if matches!(item.kind, TurnItemKind::InputTranscript { .. }))).await;
    let id = turn
        .handle()
        .append_input("external observation")
        .await
        .unwrap();
    assert!(h.h.items().iter().any(|item| item.id == id));
    assert_eq!(
        connection.command().await,
        ModelSessionCommand::Input(ModelInput::Text {
            text: "external observation".into()
        })
    );
    assert!(h.connections.try_recv().is_err());
    assert!(h.h.requests.try_recv().is_err());
    connection.close_turn().await;
    assert_eq!(finish(turn).await, TurnOutcome::Completed);
    let items = h.h.items();
    assert_eq!(
        items
            .iter()
            .filter(|item| matches!(item.kind, TurnItemKind::InputTranscript { .. }))
            .count(),
        1
    );
    assert!(!items
        .iter()
        .any(|item| matches!(item.kind, TurnItemKind::InferenceStarted { .. })));
    assert_eq!(
        ThreadProjection::from_thread(&h.h.store.0.lock().unwrap()["thread"])
            .conversation
            .iter()
            .filter(
                |message| matches!(message, ChatMessage::User { content } if content == "hello")
            )
            .count(),
        1
    );
}

#[tokio::test]
async fn tool_execution_overlaps_input_and_results_return_to_the_same_session() {
    let mut h = LiveHarness::new();
    let OpenedInteraction {
        turn,
        mut connection,
    } = h.open().await;
    connection.start("work").await;
    connection
        .item("work", "tool-item", vec![call("a", "slow")])
        .await;
    let invocation = h.h.invocation().await;
    let accepted = connection
        .observe(|kind| matches!(kind, TurnItemKind::ToolAccepted { .. }))
        .await;
    let id = accepted
        .iter()
        .find_map(|item| match &item.kind {
            TurnItemKind::ToolAccepted { operation_id, .. } => Some(operation_id.clone()),
            _ => None,
        })
        .unwrap();
    connection.end_response("work").await;
    turn.handle()
        .send_input(ModelInput::Audio { chunk: audio() })
        .await
        .unwrap();
    assert!(matches!(
        connection.command().await,
        ModelSessionCommand::Input(_)
    ));
    connection
        .emit(ModelSessionEvent::InputStarted {
            item_id: "refinement".into(),
        })
        .await;
    connection
        .emit(ModelSessionEvent::InputCommitted {
            item_id: "refinement".into(),
        })
        .await;
    connection.start("refine").await;
    let mut cancel = call("cancel", "cancel_operation");
    cancel.arguments = json!({"operation_id":id});
    connection.item("refine", "cancel-item", vec![cancel]).await;
    connection
        .observe(|kind| matches!(kind, TurnItemKind::ToolCancellationRequested { .. }))
        .await;
    assert!(invocation.context.abort_signal.is_cancelled());
    done(invocation);
    connection
        .observe(|kind| {
            matches!(
                kind,
                TurnItemKind::ToolCompleted {
                    result: ToolResult::Success { .. },
                    ..
                }
            )
        })
        .await;
    connection.end_response("refine").await;
    connection.close_turn().await;
    assert_eq!(finish(turn).await, TurnOutcome::Completed);
    assert!(h.connections.try_recv().is_err());
}

#[tokio::test]
async fn interruption_rejects_late_calls_and_final_closure_drains_audio_and_usage() {
    let mut h = LiveHarness::new();
    let OpenedInteraction {
        mut turn,
        connection,
    } = h.open().await;
    connection.start("old").await;
    connection
        .emit(ModelSessionEvent::ResponseFinished {
            response_id: "old".into(),
            status: ModelResponseStatus::Interrupted {
                reason: "new speech".into(),
            },
        })
        .await;
    connection
        .item("old", "late", vec![call("unsafe", "write")])
        .await;
    connection.start("final").await;
    connection
        .item("final", "close-item", vec![call("close", "end_turn")])
        .await;
    connection
        .emit(ModelSessionEvent::AudioDelta {
            response_id: "final".into(),
            item_id: "speech".into(),
            chunk: audio(),
        })
        .await;
    let usage = TokenUsage {
        input_tokens: 10,
        cached_input_tokens: 0,
        output_tokens: 2,
        total_tokens: 12,
    };
    for _ in 0..2 {
        connection
            .emit(ModelSessionEvent::Usage {
                event_id: "usage".into(),
                response_id: "final".into(),
                usage,
            })
            .await;
    }
    event(&mut turn, |e| {
        matches!(
            e,
            TurnEvent::Session {
                event: ModelSessionEvent::AudioDelta { .. },
                ..
            }
        )
    })
    .await;
    assert!(turn
        .handle()
        .send_input(ModelInput::Audio { chunk: audio() })
        .await
        .is_err());
    connection.end_response("final").await;
    assert_eq!(finish(turn).await, TurnOutcome::Completed);
    assert!(h.h.invocations.try_recv().is_err());
    assert_eq!(h.h.store.0.lock().unwrap()["thread"].token_usage, usage);
}

#[tokio::test]
async fn connection_loss_suspends_without_cancelling_tools_and_reconnect_restores_facts() {
    let mut h = LiveHarness::new();
    let OpenedInteraction {
        mut turn,
        connection,
    } = h.open().await;
    connection.start("work").await;
    connection
        .item("work", "tool", vec![call("a", "slow")])
        .await;
    let invocation = h.h.invocation().await;
    drop(connection);
    event(&mut turn, |e| matches!(e, TurnEvent::Suspended { .. })).await;
    assert_eq!(
        h.h.store.0.lock().unwrap()["thread"].turns[0].status,
        TurnStatus::Suspended
    );
    assert!(!invocation.context.abort_signal.is_cancelled());
    done(invocation);
    event(&mut turn, |e| matches!(e, TurnEvent::ItemCommitted { item } if matches!(item.kind, TurnItemKind::ToolCompleted { .. }))).await;
    turn.handle().reconnect_model().await.unwrap();
    let mut next = h.pending_connection().await;
    assert!(next.config.history.turns[0]
        .items
        .iter()
        .any(|item| matches!(item.kind, TurnItemKind::ToolCompleted { .. })));
    let (dummy, _) = oneshot::channel();
    std::mem::replace(&mut next.allow, dummy).send(()).unwrap();
    event(&mut turn, |e| matches!(e, TurnEvent::SessionReady { .. })).await;
    assert_eq!(
        h.h.store.0.lock().unwrap()["thread"].turns[0].status,
        TurnStatus::Running
    );
    assert!(h.h.invocations.try_recv().is_err());
    next.close_turn().await;
    assert_eq!(finish(turn).await, TurnOutcome::Completed);
}

#[tokio::test]
async fn abort_cancels_connection_establishment() {
    let mut h = LiveHarness::new();
    let turn =
        h.h.agent
            .open_stream_turn("thread", json!({}), config())
            .await
            .unwrap();
    let connection = h.pending_connection().await;
    turn.handle().abort();
    assert!(matches!(finish(turn).await, TurnOutcome::Aborted { .. }));
    assert!(connection.allow.is_closed());
}

#[tokio::test]
async fn new_speech_prevents_stale_closure_and_transport_cancel_is_explicit() {
    let mut h = LiveHarness::new();
    let OpenedInteraction {
        turn,
        mut connection,
    } = h.open().await;
    connection.start("old").await;
    connection
        .emit(ModelSessionEvent::InputStarted {
            item_id: "new-speech".into(),
        })
        .await;
    connection
        .item("old", "stale-close", vec![call("stale-close", "end_turn")])
        .await;
    connection.observe(|kind| matches!(kind, TurnItemKind::ToolOutput { call_id, result: ToolResult::Error { .. }, .. } if call_id == "stale-close")).await;
    assert_eq!(
        h.h.store.0.lock().unwrap()["thread"].turns[0].status,
        TurnStatus::Running
    );
    turn.handle().cancel_response("old").await.unwrap();
    assert_eq!(
        connection.command().await,
        ModelSessionCommand::CancelResponse {
            response_id: "old".into()
        }
    );
    connection.end_response("old").await;
    connection
        .emit(ModelSessionEvent::InputCommitted {
            item_id: "new-speech".into(),
        })
        .await;
    connection.close_turn().await;
    assert_eq!(finish(turn).await, TurnOutcome::Completed);
}

#[tokio::test]
async fn results_produced_during_reconnection_are_delivered_after_session_opens() {
    let mut h = LiveHarness::new();
    let OpenedInteraction {
        mut turn,
        connection,
    } = h.open().await;
    connection.start("work").await;
    connection
        .item("work", "tool", vec![call("a", "slow")])
        .await;
    let invocation = h.h.invocation().await;
    drop(connection);
    event(&mut turn, |e| matches!(e, TurnEvent::Suspended { .. })).await;
    turn.handle().reconnect_model().await.unwrap();
    let mut next = h.pending_connection().await;
    assert!(!next.config.history.turns[0]
        .items
        .iter()
        .any(|item| matches!(item.kind, TurnItemKind::ToolCompleted { .. })));
    done(invocation);
    event(&mut turn, |e| matches!(e, TurnEvent::ItemCommitted { item } if matches!(item.kind, TurnItemKind::ToolCompleted { .. }))).await;
    let (dummy, _) = oneshot::channel();
    std::mem::replace(&mut next.allow, dummy).send(()).unwrap();
    event(&mut turn, |e| matches!(e, TurnEvent::SessionReady { .. })).await;
    next.observe(|kind| matches!(kind, TurnItemKind::ToolCompleted { .. }))
        .await;
    next.close_turn().await;
    assert_eq!(finish(turn).await, TurnOutcome::Completed);
}

#[tokio::test]
async fn provider_responses_consume_the_turn_model_iteration_budget() {
    let mut h = LiveHarness::with_config(AgentConfig {
        turn_execution_limits: TurnExecutionLimits {
            max_model_iterations: 1,
            ..TurnExecutionLimits::default()
        },
        ..AgentConfig::default()
    });
    let OpenedInteraction { turn, connection } = h.open().await;
    connection.start("first").await;
    connection.end_response("first").await;
    connection.start("excess").await;
    assert!(matches!(finish(turn).await, TurnOutcome::Failed { .. }));
    assert_eq!(
        h.h.items()
            .iter()
            .filter(|item| matches!(item.kind, TurnItemKind::ModelResponseStarted { .. }))
            .count(),
        1
    );
}

#[tokio::test]
async fn a_blocked_transport_writer_does_not_block_model_output_or_abort() {
    let mut h = LiveHarness::new();
    let OpenedInteraction {
        mut turn,
        connection,
    } = h.open().await;
    // Fill the fake network's writer queue without reading it.
    for _ in 0..40 {
        turn.handle()
            .send_input(ModelInput::Audio { chunk: audio() })
            .await
            .unwrap();
    }
    connection.start("speaking").await;
    connection
        .emit(ModelSessionEvent::AudioDelta {
            response_id: "speaking".into(),
            item_id: "audio".into(),
            chunk: audio(),
        })
        .await;
    event(&mut turn, |e| {
        matches!(
            e,
            TurnEvent::Session {
                event: ModelSessionEvent::AudioDelta { .. },
                ..
            }
        )
    })
    .await;
    turn.handle().abort();
    assert!(matches!(finish(turn).await, TurnOutcome::Aborted { .. }));
}

#[tokio::test]
async fn streamed_call_batches_share_function_limits_and_durable_receipts() {
    let mut h = LiveHarness::with_config(AgentConfig {
        turn_execution_limits: TurnExecutionLimits {
            max_function_calls: 1,
            ..TurnExecutionLimits::default()
        },
        ..AgentConfig::default()
    });
    let OpenedInteraction {
        turn,
        mut connection,
    } = h.open().await;
    connection.start("work").await;
    connection
        .item(
            "work",
            "calls",
            vec![call("allowed", "slow"), call("excess", "write")],
        )
        .await;
    let invocation = h.h.invocation().await;
    assert_eq!(invocation.context.call_id, "allowed");
    let items = h.h.items();
    assert!(items.iter().any(|item| matches!(&item.kind, TurnItemKind::ModelItemReceived { item_id, .. } if item_id == "calls")));
    assert!(items.iter().any(|item| matches!(&item.kind, TurnItemKind::ToolOutput { call_id, result: ToolResult::Error { .. }, .. } if call_id == "excess")));
    done(invocation);
    assert!(matches!(finish(turn).await, TurnOutcome::Failed { .. }));
    assert!(h.h.invocations.try_recv().is_err());
    // Any delivered receipt is based on the same already-committed batch.
    while let Ok(command) = connection.commands.try_recv() {
        if let ModelSessionCommand::Observe { items } = command {
            assert!(items.iter().all(|item| h.h.items().contains(item)));
        }
    }
}

#[tokio::test]
async fn process_recovery_preserves_streaming_response_limits() {
    let mut h = LiveHarness::with_config(AgentConfig {
        turn_execution_limits: TurnExecutionLimits {
            max_model_iterations: 1,
            ..TurnExecutionLimits::default()
        },
        ..AgentConfig::default()
    });
    let OpenedInteraction {
        mut turn,
        connection,
    } = h.open().await;
    connection.start("first").await;
    connection.end_response("first").await;
    event(&mut turn, |e| matches!(e, TurnEvent::ItemCommitted { item } if matches!(item.kind, TurnItemKind::ModelResponseFinished { .. }))).await;
    turn.handle().abort();
    assert!(matches!(finish(turn).await, TurnOutcome::Aborted { .. }));
    let mut recovered = h.h.agent.recover_turn("thread", config()).await.unwrap();
    let mut next = h.pending_connection().await;
    let (dummy, _) = oneshot::channel();
    std::mem::replace(&mut next.allow, dummy).send(()).unwrap();
    event(&mut recovered, |e| {
        matches!(e, TurnEvent::SessionReady { .. })
    })
    .await;
    next.start("excess").await;
    assert!(matches!(
        finish(recovered).await,
        TurnOutcome::Failed { .. }
    ));
}
