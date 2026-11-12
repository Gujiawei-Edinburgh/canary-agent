use super::tests::{test_registry, TestStore};
use super::*;
use crate::model::{ModelDescriptor, ModelStreamHandler};
use crate::FunctionSpec;
use canary_agent_kernel::ChatMessage;
use serde_json::json;
use std::{num::NonZeroUsize, time::Duration};
use tokio::sync::{mpsc, oneshot};

struct Request {
    context: ModelRequest,
    reply: oneshot::Sender<Result<ModelResponse>>,
}

struct ControlledModel(mpsc::UnboundedSender<Request>);

impl ModelClient for ControlledModel {
    fn model_descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            fqn: "test/steering".into(),
            settings: json!({}),
        }
    }

    fn stream_complete<'a>(
        &'a self,
        context: ModelRequest,
        _: &'a mut ModelStreamHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
        Box::pin(async move {
            let (reply, result) = oneshot::channel();
            self.0
                .send(Request { context, reply })
                .map_err(|_| AgentError::Model("test receiver closed".into()))?;
            result
                .await
                .map_err(|_| AgentError::Model("test reply dropped".into()))?
        })
    }
}

struct Harness {
    agent: Agent,
    store: Arc<TestStore>,
    requests: mpsc::UnboundedReceiver<Request>,
}

impl Harness {
    fn new() -> Self {
        let store = Arc::new(TestStore::default());
        let (sender, requests) = mpsc::unbounded_channel();
        let agent = Agent::new(
            AgentConfig::default(),
            store.clone(),
            Arc::new(ControlledModel(sender)),
            test_registry(),
            Arc::new(crate::LocalSessionCoordinator::default()),
        );
        Self {
            agent,
            store,
            requests,
        }
    }

    fn start(&self) -> tokio::task::JoinHandle<Result<TurnOutcome>> {
        let agent = self.agent.clone();
        tokio::spawn(async move {
            agent
                .run_turn("thread", "initial request", json!({}), |_| {})
                .await
        })
    }

    async fn request(&mut self) -> Request {
        tokio::time::timeout(Duration::from_secs(2), self.requests.recv())
            .await
            .unwrap()
            .unwrap()
    }

    async fn items(&self) -> Vec<TurnItem> {
        self.store.load("thread").await.unwrap().turns[0]
            .items
            .clone()
    }
}

fn text(request: Request, text: &str) {
    request
        .reply
        .send(Ok(ModelResponse::AssistantMessage { text: text.into() }))
        .unwrap();
}

async fn finish(task: tokio::task::JoinHandle<Result<TurnOutcome>>) -> TurnOutcome {
    tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap()
}

#[tokio::test]
async fn steering_waits_for_sampling_and_preserves_the_original_response() {
    let mut h = Harness::new();
    let task = h.start();
    let first = h.request().await;
    let receipt = h.agent.steer("thread", "refinement").unwrap();
    let id = receipt.input_id.clone();
    assert!(!first.reply.is_closed());
    assert!(!h.items().await.iter().any(|item| item.id == id));
    assert!(h.requests.try_recv().is_err());
    text(first, "original snapshot answer");
    receipt.wait_recorded().await.unwrap();
    let next = h.request().await;
    assert!(next
        .context
        .messages
        .iter()
        .any(|m| matches!(m, ChatMessage::User { content } if content == "refinement")));
    assert!(h.items().await.iter().any(|item| matches!(&item.kind, TurnItemKind::ModelResponse { text:Some(text), .. } if text == "original snapshot answer")));
    text(next, "refined answer");
    assert!(
        matches!(finish(task).await, TurnOutcome::AssistantMessage { text } if text == "refined answer")
    );
    assert!(matches!(
        h.agent.steer("thread", "late"),
        Err(AgentError::TurnNotActive(_))
    ));
}

struct Invocation {
    context: FunctionContext,
    reply: oneshot::Sender<()>,
}

#[tokio::test]
async fn entire_tool_batch_finishes_before_fifo_inputs_are_recorded() {
    let mut h = Harness::new();
    let (sender, mut invocations) = mpsc::unbounded_channel();
    h.agent = h.agent.with_function(crate::SimpleFunction::new(
        FunctionSpec {
            name: "gate".into(),
            description: "test".into(),
            parameters: json!({"type":"object"}),
        },
        json!({}),
        crate::FunctionLimits {
            time_budget: Duration::from_secs(10),
            max_output_bytes: 1024,
        },
        FunctionRecoveryPolicy::Idempotent,
        move |_, context| {
            let sender = sender.clone();
            async move {
                let (reply, result) = oneshot::channel();
                sender.send(Invocation { context, reply }).unwrap();
                result.await.unwrap();
                Ok(FunctionExecution::Completed {
                    output: json!({"executed":true}),
                })
            }
        },
    ));
    let task = h.start();
    h.request()
        .await
        .reply
        .send(Ok(ModelResponse::FunctionCalls {
            calls: vec![
                ModelFunctionCall {
                    call_id: "a".into(),
                    name: "gate".into(),
                    arguments: json!({}),
                },
                ModelFunctionCall {
                    call_id: "b".into(),
                    name: "gate".into(),
                    arguments: json!({}),
                },
            ],
        }))
        .unwrap();
    let a = invocations.recv().await.unwrap();
    let first = h.agent.steer("thread", "first refinement").unwrap();
    let second = h.agent.steer("thread", "second refinement").unwrap();
    assert!(!a.context.abort_signal.is_cancelled());
    a.reply.send(()).unwrap();
    let b = invocations.recv().await.unwrap();
    assert!(!h.items().await.iter().any(|item| item.id == first.input_id));
    assert!(!b
        .context
        .projection
        .conversation
        .iter()
        .any(|m| matches!(m, ChatMessage::User { content } if content == "first refinement")));
    b.reply.send(()).unwrap();
    first.wait_recorded().await.unwrap();
    second.wait_recorded().await.unwrap();
    let next = h.request().await;
    let messages = &next.context.messages;
    let batch = messages
        .iter()
        .position(
            |m| matches!(m, ChatMessage::Assistant { tool_calls, .. } if tool_calls.len() == 2),
        )
        .unwrap();
    assert!(
        matches!(&messages[batch+1], ChatMessage::Tool { tool_call_id, .. } if tool_call_id == "a")
    );
    assert!(
        matches!(&messages[batch+2], ChatMessage::Tool { tool_call_id, .. } if tool_call_id == "b")
    );
    assert!(
        matches!(&messages[batch+3], ChatMessage::User { content } if content == "first refinement")
    );
    assert!(
        matches!(&messages[batch+4], ChatMessage::User { content } if content == "second refinement")
    );
    text(next, "done");
    finish(task).await;
}

#[tokio::test]
async fn abort_still_cancels_sampling_and_rejects_pending_receipts() {
    let mut h = Harness::new();
    let task = h.start();
    let pending = h.request().await;
    let receipt = h.agent.steer("thread", "queued").unwrap();
    assert!(!pending.reply.is_closed());
    h.agent.abort("thread").unwrap();
    assert!(matches!(finish(task).await, TurnOutcome::Aborted { .. }));
    assert!(pending.reply.is_closed());
    assert!(receipt.wait_recorded().await.is_err());
}

#[tokio::test]
async fn capacity_and_model_failure_do_not_leave_receipts_waiting_forever() {
    let mut h = Harness::new();
    h.agent = h
        .agent
        .with_steering_capacity(NonZeroUsize::new(1).unwrap());
    let task = h.start();
    let pending = h.request().await;
    let receipt = h.agent.steer("thread", "queued").unwrap();
    assert!(matches!(
        h.agent.steer("thread", "excess"),
        Err(AgentError::SteeringQueueFull { capacity: 1 })
    ));
    pending
        .reply
        .send(Err(AgentError::Model("provider failed".into())))
        .unwrap();
    assert!(matches!(finish(task).await, TurnOutcome::Failed { .. }));
    assert!(receipt.wait_recorded().await.is_err());
}

#[tokio::test]
async fn failed_persistence_never_acknowledges_a_steering_item() {
    let mut h = Harness::new();
    let task = h.start();
    let pending = h.request().await;
    let receipt = h.agent.steer("thread", "queued").unwrap();
    let mut thread = h.store.load("thread").await.unwrap();
    thread.revision = RevisionToken::from_bytes(b"another writer".to_vec());
    h.store.insert_thread(thread);
    text(pending, "done");
    assert!(tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap()
        .is_err());
    assert!(receipt.wait_recorded().await.is_err());
}

#[tokio::test]
async fn callback_steering_before_completion_forces_a_follow_up_iteration() {
    let mut h = Harness::new();
    let actor = h.agent.clone();
    let sender = h.agent.clone();
    let receipt = Arc::new(Mutex::new(None));
    let captured = receipt.clone();
    let task = tokio::spawn(async move {
        let mut first = true;
        actor
            .run_turn("thread", "initial", json!({}), move |event| {
                if first
                    && matches!(
                        event,
                        TurnStreamEvent::Model(TurnModelEvent::AssistantResponse { .. })
                    )
                {
                    first = false;
                    *captured.lock().unwrap() =
                        Some(sender.steer("thread", "at completion").unwrap());
                }
            })
            .await
    });
    text(h.request().await, "first");
    let next = h.request().await;
    let receipt = receipt.lock().unwrap().take().unwrap();
    receipt.wait_recorded().await.unwrap();
    text(next, "final");
    assert!(
        matches!(finish(task).await, TurnOutcome::AssistantMessage { text } if text == "final")
    );
}

#[tokio::test]
async fn allowed_empty_response_does_not_drop_a_queued_refinement() {
    let mut h = Harness::new();
    h.agent.config.allow_empty_response = true;
    let task = h.start();
    let first = h.request().await;
    let receipt = h.agent.steer("thread", "another requirement").unwrap();
    first
        .reply
        .send(Ok(ModelResponse::Assistant {
            continuation: None,
            text: None,
            function_calls: vec![],
        }))
        .unwrap();
    let next = h.request().await;
    receipt.wait_recorded().await.unwrap();
    text(next, "done");
    assert!(matches!(
        finish(task).await,
        TurnOutcome::AssistantMessage { .. }
    ));
}

#[tokio::test]
async fn suspension_rejects_pending_input_without_inventing_another_iteration() {
    let mut h = Harness::new();
    let sender = h.agent.clone();
    let pending = Arc::new(Mutex::new(None));
    let capture = pending.clone();
    h.agent = h.agent.with_function(crate::SimpleFunction::new(
        FunctionSpec {
            name: "approval".into(),
            description: "test".into(),
            parameters: json!({"type":"object"}),
        },
        json!({}),
        crate::FunctionLimits {
            time_budget: Duration::from_secs(2),
            max_output_bytes: 1024,
        },
        FunctionRecoveryPolicy::Idempotent,
        move |_, _| {
            let sender = sender.clone();
            let capture = capture.clone();
            async move {
                *capture.lock().unwrap() =
                    Some(sender.steer("thread", "queued during preparation").unwrap());
                Ok(FunctionExecution::SuspendedBeforeExecution {
                    suspension: Suspension {
                        id: "approval".into(),
                        kind: SuspensionKind::HumanApproval,
                        payload: json!({}),
                    },
                })
            }
        },
    ));
    let task = h.start();
    h.request()
        .await
        .reply
        .send(Ok(ModelResponse::FunctionCalls {
            calls: vec![ModelFunctionCall {
                call_id: "approve".into(),
                name: "approval".into(),
                arguments: json!({}),
            }],
        }))
        .unwrap();
    assert!(matches!(finish(task).await, TurnOutcome::Suspended { .. }));
    let receipt = pending.lock().unwrap().take().unwrap();
    assert!(receipt.wait_recorded().await.is_err());
    assert!(h.requests.try_recv().is_err());
}
