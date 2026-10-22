use super::*;
use crate::model::ModelStreamHandler;
use crate::store::ThreadContextCache;
use crate::trace::{TraceCollector, TraceEvent, TraceEventKind, TraceTurnStatus};
use crate::{
    AgentConfig, FunctionCallHook, FunctionCallHookContext, FunctionCallHookResult,
    FunctionContext, FunctionExecution, FunctionHookEventHandler, FunctionLimits,
    FunctionRecoveryPolicy, FunctionRegistry, LocalSessionCoordinator, ModelClient,
    ModelDescriptor, ModelFunctionCall, ModelRequest, ModelResponse, ModelStreamEvent,
    SimpleFunction, ThreadStore,
};
use canary_agent_kernel::{
    new_id, ChatMessage, RevisionToken, Suspension, SuspensionKind, TokenUsage, ToolResult,
    TurnItemKind,
};
use canary_agent_revision::AgentBuildRef;
use serde_json::json;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

#[derive(Default)]
struct MemoryStore(Mutex<HashMap<String, Thread>>);

#[derive(Default)]
struct RecordedTrace(Mutex<Vec<TraceEvent>>);

impl TraceCollector for RecordedTrace {
    fn record(&self, event: TraceEvent) {
        self.0.lock().unwrap().push(event);
    }
}

struct TestHook {
    name: &'static str,
    calls: Arc<Mutex<Vec<String>>>,
    fail_before: bool,
    fail_after: bool,
    store: Arc<MemoryStore>,
}

impl FunctionCallHook for TestHook {
    fn descriptor(&self) -> canary_agent_revision::ComponentRef {
        canary_agent_revision::ComponentRef::new(self.name, json!({})).unwrap()
    }

    fn before_call<'a>(
        &'a self,
        _: FunctionCallHookContext,
        _: &'a mut FunctionHookEventHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            self.calls
                .lock()
                .unwrap()
                .push(format!("before:{}", self.name));
            if self.fail_before {
                return Err(AgentError::Model("blocked by hook".into()));
            }
            Ok(())
        })
    }

    fn after_call<'a>(
        &'a self,
        context: FunctionCallHookContext,
        _: FunctionCallHookResult,
        _: &'a mut FunctionHookEventHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            assert!(self.store.0.lock().unwrap()["thread"].turns[0].items.iter().any(|item| {
                matches!(&item.kind, TurnItemKind::ToolCompleted { call_id, .. } if call_id == &context.call_id)
            }), "post-hook must observe a durable outcome");
            self.calls
                .lock()
                .unwrap()
                .push(format!("after:{}", self.name));
            if self.fail_after {
                return Err(AgentError::Model("post-hook failed".into()));
            }
            Ok(())
        })
    }
}

impl ThreadStore for MemoryStore {
    fn load<'a>(
        &'a self,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>> {
        Box::pin(async move {
            self.0
                .lock()
                .unwrap()
                .get(id)
                .cloned()
                .ok_or_else(|| AgentError::ThreadNotFound(id.into()))
        })
    }
    fn delete<'a>(&'a self, id: &'a str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            self.0.lock().unwrap().remove(id);
            Ok(())
        })
    }
    fn compare_and_commit<'a>(
        &'a self,
        mut thread: Thread,
        _: &'a crate::LeaseFence,
    ) -> Pin<Box<dyn Future<Output = Result<Thread>> + Send + 'a>> {
        Box::pin(async move {
            let mut threads = self.0.lock().unwrap();
            let current = threads
                .get(&thread.id)
                .map(|thread| thread.revision.clone())
                .unwrap_or_default();
            if current != thread.revision {
                return Err(AgentError::RevisionConflict {
                    thread_id: thread.id,
                    expected: thread.revision,
                    actual: current,
                });
            }
            thread.revision = RevisionToken::from_bytes(new_id("revision").into_bytes());
            threads.insert(thread.id.clone(), thread.clone());
            Ok(thread)
        })
    }
    fn load_context_cache<'a>(
        &'a self,
        _: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ThreadContextCache>>> + Send + 'a>> {
        Box::pin(async { Ok(None) })
    }
    fn save_context_cache<'a>(
        &'a self,
        cache: ThreadContextCache,
    ) -> Pin<Box<dyn Future<Output = Result<ThreadContextCache>> + Send + 'a>> {
        Box::pin(async move { Ok(cache) })
    }
}

struct Request {
    request: ModelRequest,
    reply: oneshot::Sender<ModelResponse>,
    deltas: mpsc::Sender<ModelStreamEvent>,
}

struct Model(mpsc::UnboundedSender<Request>);

impl ModelClient for Model {
    fn model_descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            fqn: "test/controlled".into(),
            settings: json!({}),
        }
    }
    fn stream_complete<'a>(
        &'a self,
        request: ModelRequest,
        on_event: &'a mut ModelStreamHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
        Box::pin(async move {
            let (reply, mut response) = oneshot::channel();
            let (deltas, mut events) = mpsc::channel(16);
            self.0
                .send(Request {
                    request,
                    reply,
                    deltas,
                })
                .unwrap();
            loop {
                tokio::select! {
                    event = events.recv() => if let Some(event) = event { on_event(event); },
                    response = &mut response => return response.map_err(|_| AgentError::Model("test response dropped".into())),
                }
            }
        })
    }
}

struct Invocation {
    context: FunctionContext,
    reply: oneshot::Sender<FunctionExecution>,
}

struct Harness {
    agent: Agent,
    store: Arc<MemoryStore>,
    requests: mpsc::UnboundedReceiver<Request>,
    invocations: mpsc::UnboundedReceiver<Invocation>,
}

impl Harness {
    fn new() -> Self {
        let store = Arc::new(MemoryStore::default());
        let (model, requests) = mpsc::unbounded_channel();
        let (calls, invocations) = mpsc::unbounded_channel();
        let mut registry = FunctionRegistry::new();
        for (name, policy) in [
            ("slow", FunctionRecoveryPolicy::Idempotent),
            ("write", FunctionRecoveryPolicy::NonIdempotent),
        ] {
            let calls = calls.clone();
            registry.register(SimpleFunction::new(
                crate::FunctionSpec {
                    name: name.into(),
                    description: name.into(),
                    parameters: json!({"type":"object"}),
                },
                json!({}),
                FunctionLimits {
                    time_budget: Duration::from_secs(10),
                    max_output_bytes: 4096,
                },
                policy,
                move |_args, context| {
                    let calls = calls.clone();
                    async move {
                        let (reply, response) = oneshot::channel();
                        calls.send(Invocation { context, reply }).unwrap();
                        response.await.map_err(|_| AgentError::Function {
                            name: name.into(),
                            message: "test execution disconnected".into(),
                        })
                    }
                },
            ));
        }
        let agent = Agent::new(
            AgentConfig::default(),
            store.clone(),
            Arc::new(Model(model)),
            registry,
            Arc::new(LocalSessionCoordinator::default()),
        );
        Self {
            agent,
            store,
            requests,
            invocations,
        }
    }

    async fn open(&self) -> RunningTurn {
        self.agent
            .open_turn(
                "thread",
                "original request",
                json!({"owner":"test"}),
                config(),
            )
            .await
            .unwrap()
    }

    async fn request(&mut self) -> Request {
        tokio::time::timeout(Duration::from_secs(2), self.requests.recv())
            .await
            .expect("model scheduled")
            .expect("model request")
    }

    async fn invocation(&mut self) -> Invocation {
        tokio::time::timeout(Duration::from_secs(2), self.invocations.recv())
            .await
            .expect("function scheduled")
            .expect("function invocation")
    }

    fn items(&self) -> Vec<TurnItem> {
        self.store.0.lock().unwrap()["thread"]
            .turns
            .last()
            .unwrap()
            .items
            .clone()
    }
}

fn config() -> TurnConfig {
    TurnConfig {
        cancellation_grace: Duration::from_millis(20),
        ..TurnConfig::default()
    }
}
fn call(id: &str, name: &str) -> ModelFunctionCall {
    ModelFunctionCall {
        call_id: id.into(),
        name: name.into(),
        arguments: json!({}),
    }
}
fn respond(request: Request, calls: Vec<ModelFunctionCall>) {
    request
        .reply
        .send(ModelResponse::FunctionCalls { calls })
        .unwrap();
}
fn text(request: Request) {
    request
        .reply
        .send(ModelResponse::AssistantMessage {
            text: "acknowledged".into(),
        })
        .unwrap();
}
fn done(invocation: Invocation) {
    invocation
        .reply
        .send(FunctionExecution::Completed {
            output: json!({"actual":"finished"}),
        })
        .unwrap();
}

async fn event(turn: &mut RunningTurn, predicate: impl Fn(&TurnEvent) -> bool) -> TurnEvent {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let event = turn.next_event().await.unwrap();
            if predicate(&event) {
                return event;
            }
        }
    })
    .await
    .expect("expected runtime event")
}

async fn idle(turn: &mut RunningTurn) {
    event(turn, |event| matches!(event, TurnEvent::Idle)).await;
}
async fn finish(turn: RunningTurn) -> TurnOutcome {
    tokio::time::timeout(Duration::from_secs(2), turn.wait())
        .await
        .expect("turn terminates")
        .unwrap()
}

#[tokio::test]
async fn input_supersedes_inference_and_drops_its_future() {
    let mut h = Harness::new();
    let turn = h.open().await;
    let old = h.request().await;
    turn.handle().append_input("changed my mind").await.unwrap();
    let current = h.request().await;
    assert!(old.reply.is_closed());
    assert!(current.request.messages.iter().any(
        |message| matches!(message, ChatMessage::User { content } if content == "changed my mind")
    ));
    assert!(old
        .reply
        .send(ModelResponse::FunctionCalls {
            calls: vec![call("stale", "write")]
        })
        .is_err());
    respond(current, vec![call("close", "end_turn")]);
    assert_eq!(finish(turn).await, TurnOutcome::Completed);
    assert!(!h.items().iter().any(|item| matches!(&item.kind, TurnItemKind::ModelResponse { function_calls, .. } if function_calls.iter().any(|call| call.call_id == "stale"))));
    assert!(h.invocations.try_recv().is_err());
}

#[tokio::test]
async fn explicit_closure_preserves_final_text_and_emits_committed_completion() {
    let mut h = Harness::new();
    let mut turn = h.open().await;
    h.request()
        .await
        .reply
        .send(ModelResponse::Assistant {
            continuation: None,
            text: Some("final answer".into()),
            function_calls: vec![call("close", "end_turn")],
        })
        .unwrap();
    let finished = event(&mut turn, |event| {
        matches!(event, TurnEvent::Finished { .. })
    })
    .await;
    assert_eq!(
        finished,
        TurnEvent::Finished {
            outcome: TurnOutcome::Completed
        }
    );
    assert_eq!(
        h.store.0.lock().unwrap()["thread"].turns[0].status,
        TurnStatus::Completed
    );
    assert!(h.items().iter().any(|item| matches!(&item.kind, TurnItemKind::ModelResponse { text: Some(text), function_calls, .. } if text == "final answer" && function_calls[0].name == "end_turn")));
    assert_eq!(finish(turn).await, TurnOutcome::Completed);

    let next = h
        .agent
        .open_turn(
            "thread",
            "another question",
            json!({"owner":"replacement"}),
            config(),
        )
        .await
        .unwrap();
    respond(h.request().await, vec![call("close-next", "end_turn")]);
    assert_eq!(finish(next).await, TurnOutcome::Completed);
    let threads = h.store.0.lock().unwrap();
    assert_eq!(threads["thread"].metadata, json!({"owner":"test"}));
    assert_eq!(threads["thread"].turns.len(), 2);
}

#[tokio::test]
async fn empty_response_is_either_failed_or_idle_but_never_successful_closure() {
    for allowed in [false, true] {
        let mut h = Harness::new();
        h.agent.config.allow_empty_response = allowed;
        let mut turn = h.open().await;
        respond(h.request().await, vec![]);
        if allowed {
            idle(&mut turn).await;
            assert_eq!(
                h.store.0.lock().unwrap()["thread"].turns[0].status,
                TurnStatus::Running
            );
            assert!(h.requests.try_recv().is_err());
            turn.handle().append_input("finish now").await.unwrap();
            respond(h.request().await, vec![call("close", "end_turn")]);
            assert_eq!(finish(turn).await, TurnOutcome::Completed);
        } else {
            assert!(matches!(finish(turn).await, TurnOutcome::Failed { .. }));
        }
    }
}

#[tokio::test]
async fn token_usage_and_trajectory_remain_durable_after_explicit_closure() {
    let mut h = Harness::new();
    let traces = Arc::new(RecordedTrace::default());
    h.agent = h.agent.with_shared_trace_collector(traces.clone());
    let mut turn = h.open().await;
    let request = h.request().await;
    let usage = TokenUsage {
        input_tokens: 10,
        cached_input_tokens: 5,
        output_tokens: 3,
        total_tokens: 13,
    };
    request
        .deltas
        .send(ModelStreamEvent::TokenUsage { usage })
        .await
        .unwrap();
    event(&mut turn, |event| {
        matches!(
            event,
            TurnEvent::Model {
                event: ModelStreamEvent::TokenUsage { .. },
                ..
            }
        )
    })
    .await;
    respond(request, vec![call("close", "end_turn")]);
    assert_eq!(finish(turn).await, TurnOutcome::Completed);
    assert_eq!(h.store.0.lock().unwrap()["thread"].token_usage, usage);
    let trace = traces.0.lock().unwrap();
    assert!(trace.iter().any(|event| matches!(
        event.kind,
        TraceEventKind::TurnFinished {
            status: TraceTurnStatus::Completed
        }
    )));
    assert!(trace.iter().any(|event| matches!(&event.kind, TraceEventKind::InteractionItem { item } if matches!(item.kind, TurnItemKind::UserInput { .. }))));
    assert!(trace
        .windows(2)
        .all(|pair| pair[0].sequence < pair[1].sequence));
}

#[tokio::test]
async fn hooks_unwind_only_entered_hooks_and_post_failures_do_not_fail_the_turn() {
    for blocked in [false, true] {
        let mut h = Harness::new();
        let calls = Arc::new(Mutex::new(Vec::new()));
        for (name, fail_before) in [("first", false), ("second", blocked), ("third", false)] {
            h.agent = h.agent.with_function_call_hook(TestHook {
                name,
                calls: calls.clone(),
                fail_before,
                fail_after: name == "first",
                store: h.store.clone(),
            });
        }
        let mut turn = h.open().await;
        respond(h.request().await, vec![call("tool", "slow")]);
        if !blocked {
            done(h.invocation().await);
        }
        let request = h.request().await;
        event(&mut turn, |event| matches!(event, TurnEvent::Hook { event } if event.message == "post-hook failed")).await;
        respond(request, vec![call("close", "end_turn")]);
        assert_eq!(finish(turn).await, TurnOutcome::Completed);
        let expected = if blocked {
            vec!["before:first", "before:second", "after:first"]
        } else {
            vec![
                "before:first",
                "before:second",
                "before:third",
                "after:third",
                "after:second",
                "after:first",
            ]
        };
        assert_eq!(*calls.lock().unwrap(), expected);
        let result = h
            .items()
            .into_iter()
            .find_map(|item| match item.kind {
                TurnItemKind::ToolCompleted {
                    call_id, result, ..
                } if call_id == "tool" => Some(result),
                _ => None,
            })
            .expect("durable tool result");
        assert_eq!(matches!(result, ToolResult::Error { .. }), blocked);
        assert!(h.invocations.try_recv().is_err());
    }
}

#[test]
fn spec_snapshot_still_describes_the_live_agent() {
    let h = Harness::new();
    let snapshot = h.agent.spec_snapshot(AgentBuildRef {
        id: "application-build".into(),
    });
    assert_eq!(snapshot.build.id, "application-build");
    assert_eq!(snapshot.model.fqn, "test/controlled");
    assert_eq!(snapshot.prompts.system, h.agent.config.system_prompt);
    assert!(snapshot.tools.contains_key("slow"));
    assert_eq!(
        snapshot.tools["slow"].interface.parameters,
        json!({"type":"object"})
    );
}

#[tokio::test]
async fn streaming_is_interruptible_and_text_only_leaves_turn_open() {
    let mut h = Harness::new();
    let mut turn = h.open().await;
    let first = h.request().await;
    first
        .deltas
        .send(ModelStreamEvent::AssistantDelta {
            text: "partial".into(),
        })
        .await
        .unwrap();
    event(&mut turn, |event| {
        matches!(
            event,
            TurnEvent::Model {
                event: ModelStreamEvent::AssistantDelta { .. },
                ..
            }
        )
    })
    .await;
    turn.handle().append_input("interrupt").await.unwrap();
    event(&mut turn, |event| matches!(event, TurnEvent::ItemCommitted { item } if matches!(item.kind, TurnItemKind::InferenceInterrupted { .. }))).await;
    text(h.request().await);
    idle(&mut turn).await;
    assert_eq!(
        h.store.0.lock().unwrap()["thread"].turns[0].status,
        TurnStatus::Running
    );
    turn.handle().append_input("done now").await.unwrap();
    respond(h.request().await, vec![call("close", "end_turn")]);
    finish(turn).await;
}

#[tokio::test]
async fn new_input_can_cancel_a_running_operation_without_rewriting_acceptance() {
    let mut h = Harness::new();
    let mut turn = h.open().await;
    respond(h.request().await, vec![call("a", "slow")]);
    let mut invocation = h.invocation().await;
    let accepted = h
        .items()
        .into_iter()
        .find(|item| matches!(item.kind, TurnItemKind::ToolAccepted { .. }))
        .unwrap();
    let TurnItemKind::ToolAccepted { operation_id, .. } = &accepted.kind else {
        unreachable!()
    };
    turn.handle()
        .append_input("stop that operation")
        .await
        .unwrap();
    let request = h.request().await;
    assert!(request.request.messages.iter().any(|message| matches!(message, ChatMessage::Tool { tool_call_id, content, .. } if tool_call_id == "a" && content["status"] == "accepted")));
    let mut cancel = call("b", "cancel_operation");
    cancel.arguments = json!({"operation_id":operation_id});
    respond(request, vec![cancel]);
    invocation.context.abort_signal.cancelled().await;
    text(h.request().await);
    idle(&mut turn).await;
    done(invocation);
    let request = h.request().await;
    assert!(request.request.messages.iter().any(|message| matches!(message, ChatMessage::User { content } if content.contains("tool_completed") && content.contains(operation_id))));
    assert_eq!(request.request.messages.iter().filter(|message| matches!(message, ChatMessage::Tool { tool_call_id, .. } if tool_call_id == "a")).count(), 1);
    respond(request, vec![call("close", "end_turn")]);
    finish(turn).await;
    assert!(h.items().contains(&accepted));
    let items = h.items();
    let input = items.iter().position(|item| matches!(&item.kind, TurnItemKind::UserInput { text, .. } if text == "stop that operation")).unwrap();
    let completion = items
        .iter()
        .position(|item| matches!(item.kind, TurnItemKind::ToolCompleted { .. }))
        .unwrap();
    assert!(input < completion);
}

#[tokio::test]
async fn end_turn_rejects_active_operations() {
    let mut h = Harness::new();
    let mut turn = h.open().await;
    respond(h.request().await, vec![call("a", "slow")]);
    let invocation = h.invocation().await;
    turn.handle().append_input("close").await.unwrap();
    respond(h.request().await, vec![call("early_close", "end_turn")]);
    let rejection = h.request().await;
    assert!(rejection.request.messages.iter().any(|message| matches!(message, ChatMessage::Tool { tool_call_id, content, .. } if tool_call_id == "early_close" && content.as_str().is_some_and(|text| text.contains("cannot close")))));
    text(rejection);
    idle(&mut turn).await;
    done(invocation);
    respond(h.request().await, vec![call("close", "end_turn")]);
    finish(turn).await;
}

#[tokio::test]
async fn unseen_completion_prevents_stale_closure() {
    let mut h = Harness::new();
    let mut turn = h.open().await;
    respond(h.request().await, vec![call("a", "slow")]);
    let invocation = h.invocation().await;
    turn.handle().append_input("are we done?").await.unwrap();
    let old_view = h.request().await;
    done(invocation);
    event(&mut turn, |event| matches!(event, TurnEvent::ItemCommitted { item } if matches!(item.kind, TurnItemKind::ToolCompleted { .. }))).await;
    respond(old_view, vec![call("stale_close", "end_turn")]);
    let fresh = h.request().await;
    assert!(fresh.request.messages.iter().any(|message| matches!(message, ChatMessage::Tool { tool_call_id, content, .. } if tool_call_id == "stale_close" && content.as_str().is_some_and(|text| text.contains("cannot close")))));
    respond(fresh, vec![call("close", "end_turn")]);
    finish(turn).await;
}

#[tokio::test]
async fn batches_have_all_receipts_before_next_model_request() {
    let mut h = Harness::new();
    let turn = h.open().await;
    respond(
        h.request().await,
        vec![call("a", "slow"), call("b", "slow")],
    );
    let a = h.invocation().await;
    let b = h.invocation().await;
    turn.handle()
        .append_input("while both are running")
        .await
        .unwrap();
    let request = h.request().await;
    let index = request.request.messages.iter().position(|message| matches!(message, ChatMessage::Assistant { tool_calls, .. } if tool_calls.len() == 2)).unwrap();
    assert!(
        matches!(&request.request.messages[index + 1], ChatMessage::Tool { tool_call_id, .. } if tool_call_id == "a")
    );
    assert!(
        matches!(&request.request.messages[index + 2], ChatMessage::Tool { tool_call_id, .. } if tool_call_id == "b")
    );
    text(request);
    turn.handle().abort();
    done(a);
    done(b);
    assert!(matches!(finish(turn).await, TurnOutcome::Aborted { .. }));
}

#[tokio::test]
async fn abort_propagates_to_tool_and_preserves_its_actual_outcome() {
    let mut h = Harness::new();
    let turn = h.open().await;
    respond(h.request().await, vec![call("a", "slow")]);
    let mut invocation = h.invocation().await;
    h.agent.abort("thread").unwrap();
    invocation.context.abort_signal.cancelled().await;
    done(invocation);
    assert!(matches!(finish(turn).await, TurnOutcome::Aborted { .. }));
    assert!(h.items().iter().any(|item| matches!(&item.kind, TurnItemKind::ToolCompleted { result: ToolResult::Success { output }, .. } if output["actual"] == "finished")));
}

#[tokio::test]
async fn uncooperative_cancellation_requires_reconciliation() {
    let mut h = Harness::new();
    let mut turn = h.open().await;
    respond(h.request().await, vec![call("a", "slow")]);
    let invocation = h.invocation().await;
    let accepted = h
        .items()
        .into_iter()
        .find_map(|item| match item.kind {
            TurnItemKind::ToolAccepted { operation_id, .. } => Some(operation_id),
            _ => None,
        })
        .unwrap();
    turn.handle().cancel_operation(&accepted).await.unwrap();
    event(&mut turn, |event| matches!(event, TurnEvent::ItemCommitted { item } if matches!(item.kind, TurnItemKind::ToolIndeterminate { .. }))).await;
    assert!(invocation.reply.is_closed());
    text(h.request().await);
    idle(&mut turn).await;
    turn.handle()
        .resolve_operation(
            &accepted,
            SuspensionResolution::ExternalResult {
                output: json!({"reconciled":true}),
            },
        )
        .await
        .unwrap();
    respond(h.request().await, vec![call("close", "end_turn")]);
    finish(turn).await;
}

#[tokio::test]
async fn pre_execution_suspension_can_resume_without_closing_interaction() {
    let mut h = Harness::new();
    let mut turn = h.open().await;
    respond(h.request().await, vec![call("a", "slow")]);
    h.invocation()
        .await
        .reply
        .send(FunctionExecution::SuspendedBeforeExecution {
            suspension: Suspension {
                id: "approval".into(),
                kind: SuspensionKind::HumanApproval,
                payload: json!({}),
            },
        })
        .unwrap();
    text(h.request().await);
    idle(&mut turn).await;
    let id = h
        .items()
        .into_iter()
        .find_map(|item| match item.kind {
            TurnItemKind::ToolAccepted { operation_id, .. } => Some(operation_id),
            _ => None,
        })
        .unwrap();
    turn.handle()
        .resolve_operation(&id, SuspensionResolution::Approve)
        .await
        .unwrap();
    done(h.invocation().await);
    respond(h.request().await, vec![call("close", "end_turn")]);
    finish(turn).await;
    assert_eq!(
        h.items()
            .iter()
            .filter(|item| matches!(item.kind, TurnItemKind::ToolAccepted { .. }))
            .count(),
        1
    );
}

#[tokio::test]
async fn recovery_retries_only_idempotent_started_operations() {
    let mut h = Harness::new();
    let mut thread = Thread::new("thread");
    let mut turn = Turn::new();
    turn.push_item(TurnItem::new(
        TurnItemSource::Runtime,
        TurnItemKind::TurnOpened,
    ));
    let calls = vec![call("read", "slow"), call("write", "write")];
    turn.push_item(TurnItem::new(
        TurnItemSource::Model,
        TurnItemKind::ModelResponse {
            continuation: None,
            text: None,
            function_calls: calls.clone(),
        },
    ));
    for call in calls {
        turn.push_item(TurnItem::new(
            TurnItemSource::Runtime,
            TurnItemKind::ToolAccepted {
                operation_id: call.call_id.clone(),
                call_id: call.call_id.clone(),
                name: call.name.clone(),
            },
        ));
        turn.push_item(TurnItem::new(
            TurnItemSource::Runtime,
            TurnItemKind::FunctionCallStarted {
                call_id: call.call_id,
                name: call.name,
                attempt: 1,
            },
        ));
    }
    thread.turns.push(turn);
    h.store.0.lock().unwrap().insert("thread".into(), thread);
    let mut turn = h.agent.recover_turn("thread", config()).await.unwrap();
    let invocation = h.invocation().await;
    assert_eq!(invocation.context.call_id, "read");
    text(h.request().await);
    idle(&mut turn).await;
    assert!(h.invocations.try_recv().is_err());
    assert!(h.items().iter().any(|item| matches!(&item.kind, TurnItemKind::ToolIndeterminate { operation_id, .. } if operation_id == "write")));
    turn.handle()
        .resolve_operation(
            "write",
            SuspensionResolution::ExternalResult {
                output: json!({"reconciled":true}),
            },
        )
        .await
        .unwrap();
    text(h.request().await);
    idle(&mut turn).await;
    done(invocation);
    respond(h.request().await, vec![call("close", "end_turn")]);
    finish(turn).await;
}

#[tokio::test]
async fn function_limit_executes_permitted_call_and_records_all_excess_calls() {
    let mut h = Harness::new();
    h.agent.config.turn_execution_limits.max_function_calls = 1;
    let turn = h.open().await;
    respond(
        h.request().await,
        vec![call("allowed", "slow"), call("excess", "write")],
    );
    let invocation = h.invocation().await;
    assert_eq!(invocation.context.call_id, "allowed");
    done(invocation);
    assert!(matches!(finish(turn).await, TurnOutcome::Failed { .. }));
    assert!(h.invocations.try_recv().is_err());
    assert!(h.items().iter().any(|item| matches!(&item.kind, TurnItemKind::ToolOutput { call_id, result: ToolResult::Error { .. }, .. } if call_id == "excess")));
    assert!(h.items().iter().any(|item| matches!(&item.kind, TurnItemKind::ToolCompleted { call_id, result: ToolResult::Success { .. }, .. } if call_id == "allowed")));
}

#[tokio::test]
async fn progress_is_optional_and_does_not_schedule_inference() {
    let mut h = Harness::new();
    let mut turn = h.open().await;
    respond(h.request().await, vec![call("a", "slow")]);
    let invocation = h.invocation().await;
    let id = h
        .items()
        .into_iter()
        .find_map(|item| match item.kind {
            TurnItemKind::ToolAccepted { operation_id, .. } => Some(operation_id),
            _ => None,
        })
        .unwrap();
    turn.handle()
        .report_progress(&id, json!({"step":2}))
        .await
        .unwrap();
    event(&mut turn, |event| matches!(event, TurnEvent::ItemCommitted { item } if matches!(item.kind, TurnItemKind::ToolProgress { .. }))).await;
    assert!(h.requests.try_recv().is_err());
    turn.handle().append_input("status?").await.unwrap();
    let request = h.request().await;
    assert!(request.request.messages.iter().any(|message| matches!(message, ChatMessage::User { content } if content.contains("active_operations") && content.contains("step"))));
    text(request);
    idle(&mut turn).await;
    done(invocation);
    respond(h.request().await, vec![call("close", "end_turn")]);
    finish(turn).await;
}

#[tokio::test]
async fn iteration_limit_also_applies_to_restarted_inferences() {
    let mut h = Harness::new();
    h.agent.config.turn_execution_limits.max_model_iterations = 1;
    let turn = h.open().await;
    let pending = h.request().await;
    turn.handle().append_input("restart").await.unwrap();
    assert!(matches!(finish(turn).await, TurnOutcome::Failed { .. }));
    assert!(pending.reply.is_closed());
    assert!(h.requests.try_recv().is_err());
}

#[tokio::test]
async fn operation_panic_is_indeterminate_and_does_not_strand_the_interaction() {
    let mut h = Harness::new();
    h.agent = h.agent.with_function(SimpleFunction::new(
        crate::FunctionSpec {
            name: "panic".into(),
            description: "test panic".into(),
            parameters: json!({}),
        },
        json!({}),
        FunctionLimits {
            time_budget: Duration::from_secs(1),
            max_output_bytes: 128,
        },
        FunctionRecoveryPolicy::NonIdempotent,
        |_, _| async {
            panic!("executor lost");
            #[allow(unreachable_code)]
            Ok(FunctionExecution::Completed {
                output: Value::Null,
            })
        },
    ));
    let mut turn = h.open().await;
    respond(h.request().await, vec![call("a", "panic")]);
    event(&mut turn, |event| matches!(event, TurnEvent::ItemCommitted { item } if matches!(item.kind, TurnItemKind::ToolIndeterminate { .. }))).await;
    text(h.request().await);
    idle(&mut turn).await;
    turn.handle().abort();
    assert!(matches!(finish(turn).await, TurnOutcome::Aborted { .. }));
}
