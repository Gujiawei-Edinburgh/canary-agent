//! Example composition of a voice actor and a persistent request-based worker.
use crate::tools;
use canary_agent_kernel::{TurnItemKind, new_id};
use canary_agent_observability::JsonlTraceCollector;
use canary_agent_openai::{ChatCompletionsClient, ModelConfig};
use canary_agent_runtime::{
    Agent, AgentConfig, AgentError, FunctionExecution, FunctionLimits, FunctionRecoveryPolicy,
    FunctionRegistry, FunctionSpec, LocalSessionCoordinator, Result, SimpleFunction, TurnConfig,
    TurnEvent, TurnHandle,
};
use canary_agent_store_json::JsonFileThreadStore;
use serde_json::json;
use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::Duration,
};
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct Bridge {
    worker: Agent,
    thread_id: String,
    handle: Arc<Mutex<Option<TurnHandle>>>,
    actor: Arc<OnceLock<TurnHandle>>,
    task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl Bridge {
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }
    pub fn new(
        store: Arc<JsonFileThreadStore>,
        trace: Arc<JsonlTraceCollector>,
        workspace: PathBuf,
    ) -> Result<Self> {
        let model = std::env::var("CANARY_AGENT_MODEL")
            .map_err(|_| AgentError::Model("set CANARY_AGENT_MODEL for the live worker".into()))?;
        let api_key = std::env::var("CANARY_AGENT_API_KEY").map_err(|_| {
            AgentError::Model("set CANARY_AGENT_API_KEY for the live worker".into())
        })?;
        let client = ChatCompletionsClient::new(ModelConfig {
            model,
            api_key,
            base_url: std::env::var("CANARY_AGENT_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".into()),
            reasoning_effort: std::env::var("CANARY_AGENT_REASONING_EFFORT")
                .unwrap_or_else(|_| ModelConfig::default_reasoning_effort()),
        });
        let worker = Agent::new(
            AgentConfig { system_prompt: "You are a worker agent assisting a voice actor. \
            Execute delegated tasks using tools. Later inputs may be refinements or new requests; distinguish them from context. \
            Tool acceptance is not completion; results arrive later. Do not issue sleep or polling commands to wait for tools. \
            Return concise factual findings to the actor. Do not call end_turn unless explicitly asked to end the worker interaction.".into(),
                ..AgentConfig::default() },
            store,
            Arc::new(client),
            tools::registry(workspace),
            Arc::new(LocalSessionCoordinator::default())).with_shared_trace_collector(trace);
        Ok(Self {
            worker,
            thread_id: new_id("live_worker"),
            handle: Arc::new(Mutex::new(None)),
            actor: Arc::new(OnceLock::new()),
            task: Arc::new(Mutex::new(None)),
        })
    }

    pub fn attach(&self, actor: TurnHandle) {
        let _ = self.actor.set(actor);
    }

    pub fn registry(&self) -> FunctionRegistry {
        let mut registry = FunctionRegistry::new();
        let bridge = self.clone();
        registry.register(SimpleFunction::new(FunctionSpec {
            name: "delegate".into(), description: "Send a task or refinement to the persistent worker. Acceptance is not completion; results arrive later.".into(),
            parameters: json!({"type":"object","properties":{"message":{"type":"string"}},"required":["message"],"additionalProperties":false}),
        }, json!({"type":"object"}), FunctionLimits { time_budget: Duration::from_secs(30), max_output_bytes: 4096 }, FunctionRecoveryPolicy::NonIdempotent,
        move |args: serde_json::Value, _: canary_agent_runtime::FunctionContext| { let bridge = bridge.clone(); async move {
            let message = args.get("message").and_then(|v| v.as_str()).filter(|s| !s.trim().is_empty()).ok_or_else(|| AgentError::Model("delegate requires a nonempty message".into()))?;
            bridge.submit(message).await?;
            Ok(FunctionExecution::Completed { output: json!({"status":"accepted", "worker_thread":bridge.thread_id}) })
        }}));
        registry
    }

    async fn submit(&self, message: &str) -> Result<()> {
        println!("\n[worker delegation] {message}");
        let mut handle = self.handle.lock().await;
        if let Some(worker) = handle.as_ref() {
            return worker.append_input(message).await.map(|_| ());
        }
        let actor = self
            .actor
            .get()
            .cloned()
            .ok_or_else(|| AgentError::Model("actor not ready".into()))?;
        let mut turn = self
            .worker
            .open_turn(
                &self.thread_id,
                message,
                json!({"application":"live_worker"}),
                TurnConfig::default(),
            )
            .await?;
        *handle = Some(turn.handle());
        let slot = self.handle.clone();
        *self.task.lock().await = Some(tokio::spawn(async move {
            while let Ok(event) = turn.next_event().await {
                match event {
                    TurnEvent::ItemCommitted { item } => {
                        if let TurnItemKind::ModelResponse {
                            text: Some(text),
                            function_calls,
                            ..
                        } = &item.kind
                            && !text.trim().is_empty()
                        {
                            println!("\n[worker] {text}");
                            if function_calls.is_empty()
                                && let Err(error) =
                                    actor.append_input(format!("Worker response: {text}")).await
                            {
                                eprintln!("worker response delivery failed: {error}");
                            }
                        } else if let TurnItemKind::ToolAccepted {
                            name, operation_id, ..
                        } = &item.kind
                        {
                            println!("\n[worker tool] {name}: {operation_id}");
                        } else if let TurnItemKind::ToolCompleted { name, result, .. } = &item.kind
                        {
                            println!("\n[worker tool completed] {name}: {result:?}");
                        }
                    }
                    TurnEvent::Finished { outcome } => {
                        if let canary_agent_runtime::TurnOutcome::Failed { error } = outcome {
                            eprintln!("worker failed: {error}");
                            if let Err(delivery_error) = actor
                                .append_input(format!(
                                    "Worker failed; the task was not completed. Error: {error}"
                                ))
                                .await
                            {
                                eprintln!("worker failure delivery failed: {delivery_error}");
                            }
                        }
                        break;
                    }
                    _ => {}
                }
            }
            *slot.lock().await = None;
            let _ = turn.wait().await;
        }));
        Ok(())
    }

    pub async fn shutdown(&self) {
        if let Some(handle) = self.handle.lock().await.as_ref() {
            handle.abort();
        }
        if let Some(task) = self.task.lock().await.take() {
            let _ = task.await;
        }
    }
}
