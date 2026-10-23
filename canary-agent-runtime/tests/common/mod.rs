// Shared fixtures are compiled independently by each integration test suite.
#![allow(dead_code)]
use canary_agent_kernel::{new_id, RevisionToken, Thread, TurnItem};
use canary_agent_runtime::model::ModelStreamHandler;
use canary_agent_runtime::*;
use serde_json::json;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

#[derive(Default)]
pub struct MemoryStore(pub Mutex<HashMap<String, Thread>>);

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
        _: &'a canary_agent_runtime::LeaseFence,
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

pub struct Request {
    pub request: ModelRequest,
    pub reply: oneshot::Sender<ModelResponse>,
    pub deltas: mpsc::Sender<ModelStreamEvent>,
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

pub struct Invocation {
    pub context: FunctionContext,
    pub reply: oneshot::Sender<FunctionExecution>,
}

pub struct Harness {
    pub agent: Agent,
    pub store: Arc<MemoryStore>,
    pub requests: mpsc::UnboundedReceiver<Request>,
    pub invocations: mpsc::UnboundedReceiver<Invocation>,
}

impl Harness {
    pub fn new() -> Self {
        Self::with_config(AgentConfig::default())
    }

    pub fn with_config(config: AgentConfig) -> Self {
        Self::with_model(config, None)
    }

    pub fn with_model(config: AgentConfig, client: Option<Arc<dyn ModelClient>>) -> Self {
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
                canary_agent_runtime::FunctionSpec {
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
            config,
            store.clone(),
            client.unwrap_or_else(|| Arc::new(Model(model))),
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

    pub async fn open(&self) -> RunningTurn {
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

    pub async fn request(&mut self) -> Request {
        tokio::time::timeout(Duration::from_secs(2), self.requests.recv())
            .await
            .expect("model scheduled")
            .expect("model request")
    }

    pub async fn invocation(&mut self) -> Invocation {
        tokio::time::timeout(Duration::from_secs(2), self.invocations.recv())
            .await
            .expect("function scheduled")
            .expect("function invocation")
    }

    pub fn items(&self) -> Vec<TurnItem> {
        self.store.0.lock().unwrap()["thread"]
            .turns
            .last()
            .unwrap()
            .items
            .clone()
    }
}

pub fn config() -> TurnConfig {
    TurnConfig {
        cancellation_grace: Duration::from_millis(20),
        ..TurnConfig::default()
    }
}
pub fn call(id: &str, name: &str) -> ModelFunctionCall {
    ModelFunctionCall {
        call_id: id.into(),
        name: name.into(),
        arguments: json!({}),
    }
}
pub fn respond(request: Request, calls: Vec<ModelFunctionCall>) {
    request
        .reply
        .send(ModelResponse::FunctionCalls { calls })
        .unwrap();
}
pub fn text(request: Request) {
    request
        .reply
        .send(ModelResponse::AssistantMessage {
            text: "acknowledged".into(),
        })
        .unwrap();
}
pub fn done(invocation: Invocation) {
    invocation
        .reply
        .send(FunctionExecution::Completed {
            output: json!({"actual":"finished"}),
        })
        .unwrap();
}

pub async fn event(turn: &mut RunningTurn, predicate: impl Fn(&TurnEvent) -> bool) -> TurnEvent {
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

pub async fn idle(turn: &mut RunningTurn) {
    event(turn, |event| matches!(event, TurnEvent::Idle)).await;
}
pub async fn finish(turn: RunningTurn) -> TurnOutcome {
    tokio::time::timeout(Duration::from_secs(2), turn.wait())
        .await
        .expect("turn terminates")
        .unwrap()
}
