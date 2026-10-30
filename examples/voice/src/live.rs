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
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::Duration,
};
use tokio::sync::Mutex;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DelegateRequest {
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delegation_arguments_are_strict_and_nonempty() {
        assert_eq!(
            DelegateRequest::parse(json!({"message": "search the web"}))
                .unwrap()
                .message,
            "search the web"
        );
        for value in [
            json!({}),
            json!({"message": "  "}),
            json!({"message": 1}),
            json!({"message": "x", "tool": "exec_command"}),
        ] {
            assert!(DelegateRequest::parse(value).is_err());
        }
    }

    #[test]
    fn receipt_represents_acceptance_not_completion() {
        let receipt = DelegateAck {
            status: DelegateStatus::Accepted,
            delegation_id: "call-1".into(),
            worker_thread: "worker-1".into(),
        };
        assert_eq!(
            serde_json::to_value(receipt).unwrap(),
            json!({"status": "accepted", "delegation_id": "call-1", "worker_thread": "worker-1"})
        );
    }

    #[test]
    fn worker_report_preserves_multiple_delegations_and_artifacts() {
        let value = json!({
            "delegation_ids": ["call-1", "call-2"],
            "status": "partial",
            "summary": "Reviewed cancellation",
            "artifacts": [{"kind": "file", "path": "ARCHITECTURE.md", "description": "Report"}],
            "sources": [{"title": "Reference", "url": "https://example.com"}],
            "limitations": ["Review incomplete"]
        });
        let report: WorkerReport = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(report).unwrap(), value);
        assert!(
            serde_json::from_value::<WorkerReport>(
                json!({"delegation_ids": ["call-1"], "status": "completed"})
            )
            .is_err()
        );
    }
}

impl DelegateRequest {
    fn parse(value: Value) -> Result<Self> {
        let request: Self = serde_json::from_value(value)
            .map_err(|error| AgentError::Model(format!("invalid delegate arguments: {error}")))?;
        if request.message.trim().is_empty() {
            return Err(AgentError::Model(
                "delegate message must not be empty".into(),
            ));
        }
        Ok(request)
    }
}

#[derive(Debug, Serialize)]
struct DelegateAck {
    status: DelegateStatus,
    delegation_id: String,
    worker_thread: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReportStatus {
    Completed,
    Partial,
    Failed,
    NeedsInput,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    kind: String,
    path: String,
    description: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    title: String,
    url: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerReport {
    delegation_ids: Vec<String>,
    status: ReportStatus,
    summary: String,
    artifacts: Vec<Artifact>,
    sources: Vec<Source>,
    limitations: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum DelegateStatus {
    Accepted,
}

fn register_report_tool(
    registry: &mut FunctionRegistry,
    actor: Arc<OnceLock<TurnHandle>>,
    known_ids: Arc<Mutex<HashSet<String>>>,
) {
    registry.register(SimpleFunction::new(
        FunctionSpec {
            name: "report_result".into(),
            description: "Report a task outcome to the voice actor. Include only delegation IDs addressed by this report.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "delegation_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                    "status": {"type": "string", "enum": ["completed", "partial", "failed", "needs_input"]},
                    "summary": {"type": "string", "minLength": 1},
                    "artifacts": {"type": "array", "items": {
                        "type": "object", "properties": {"kind": {"type": "string"}, "path": {"type": "string"}, "description": {"type": "string"}},
                        "required": ["kind", "path", "description"], "additionalProperties": false
                    }},
                    "sources": {"type": "array", "items": {
                        "type": "object", "properties": {"title": {"type": "string"}, "url": {"type": "string"}},
                        "required": ["title", "url"], "additionalProperties": false
                    }},
                    "limitations": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["delegation_ids", "status", "summary", "artifacts", "sources", "limitations"],
                "additionalProperties": false
            }),
        },
        json!({"type": "object", "properties": {"status": {"type": "string", "enum": ["delivered"]}}, "required": ["status"], "additionalProperties": false}),
        FunctionLimits { time_budget: Duration::from_secs(300), max_output_bytes: 4096 * 1024 },
        FunctionRecoveryPolicy::NonIdempotent,
        move |args: Value, _: canary_agent_runtime::FunctionContext| {
            let actor = actor.clone();
            let known_ids = known_ids.clone();
            async move {
                let report: WorkerReport = serde_json::from_value(args)
                    .map_err(|error| AgentError::Model(format!("invalid worker report: {error}")))?;
                if report.delegation_ids.is_empty() || report.summary.trim().is_empty() {
                    return Err(AgentError::Model("report requires delegation IDs and a nonempty summary".into()));
                }
                let known = known_ids.lock().await;
                if report.delegation_ids.iter().any(|id| !known.contains(id)) {
                    return Err(AgentError::Model("report contains an unknown delegation ID".into()));
                }
                drop(known);
                let actor = actor.get().ok_or_else(|| AgentError::Model("actor not ready".into()))?;
                let text = serde_json::to_string(&report)?;
                actor.append_input(format!("Worker report: {text}")).await?;
                Ok(FunctionExecution::Completed { output: json!({"status": "delivered"}) })
            }
        },
    ));
}

#[derive(Clone)]
pub struct Bridge {
    worker: Agent,
    thread_id: String,
    handle: Arc<Mutex<Option<TurnHandle>>>,
    actor: Arc<OnceLock<TurnHandle>>,
    task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    delegation_ids: Arc<Mutex<HashSet<String>>>,
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
        let actor = Arc::new(OnceLock::<TurnHandle>::new());
        let delegation_ids = Arc::new(Mutex::new(HashSet::new()));
        let mut registry = tools::registry(workspace);
        register_report_tool(&mut registry, actor.clone(), delegation_ids.clone());
        let worker = Agent::new(
            AgentConfig {
                system_prompt: concat!(
                    "You are a worker agent assisting a voice actor. ",
                    "Execute delegated tasks using tools. Later inputs may be refinements ",
                    "or new requests; distinguish them from context. Tool acceptance is ",
                    "not completion; results arrive later. Do not issue sleep or polling ",
                    "commands to wait for tools. Submit findings using report_result with ",
                    "the delegation IDs addressed. Report artifacts only after verifying ",
                    "them. Do not call end_turn unless asked to end the interaction."
                )
                .into(),
                ..AgentConfig::default()
            },
            store,
            Arc::new(client),
            registry,
            Arc::new(LocalSessionCoordinator::default()),
        )
        .with_shared_trace_collector(trace);
        Ok(Self {
            worker,
            thread_id: new_id("live_worker"),
            handle: Arc::new(Mutex::new(None)),
            actor,
            task: Arc::new(Mutex::new(None)),
            delegation_ids,
        })
    }

    pub fn attach(&self, actor: TurnHandle) {
        let _ = self.actor.set(actor);
    }

    pub fn registry(&self) -> FunctionRegistry {
        let mut registry = FunctionRegistry::new();
        let bridge = self.clone();
        registry.register(SimpleFunction::new(
            FunctionSpec {
                name: "delegate".into(),
                description: concat!(
                    "Submit a task or refinement to the worker, including relevant context. ",
                    "The worker chooses its tools. Acceptance is not completion; ",
                    "worker responses arrive separately."
                )
                .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Task or refinement with the relevant user context."
                        }
                    },
                    "required": ["message"],
                    "additionalProperties": false
                }),
            },
            json!({
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["accepted"]},
                    "delegation_id": {"type": "string"},
                    "worker_thread": {"type": "string"}
                },
                "required": ["status", "delegation_id", "worker_thread"],
                "additionalProperties": false
            }),
            FunctionLimits {
                time_budget: Duration::from_secs(30),
                max_output_bytes: 4096,
            },
            FunctionRecoveryPolicy::NonIdempotent,
            move |args: Value, context: canary_agent_runtime::FunctionContext| {
                let bridge = bridge.clone();
                async move {
                    let request = DelegateRequest::parse(args)?;
                    bridge.submit(&context.call_id, &request.message).await?;
                    let ack = DelegateAck {
                        status: DelegateStatus::Accepted,
                        delegation_id: context.call_id,
                        worker_thread: bridge.thread_id.clone(),
                    };
                    Ok(FunctionExecution::Completed {
                        output: serde_json::to_value(ack)?,
                    })
                }
            },
        ));
        registry
    }

    async fn submit(&self, delegation_id: &str, message: &str) -> Result<()> {
        println!("\n[worker delegation] {message}");
        self.delegation_ids
            .lock()
            .await
            .insert(delegation_id.into());
        let input = json!({"delegation_id": delegation_id, "message": message}).to_string();
        let result = self.submit_input(&input).await;
        if result.is_err() {
            self.delegation_ids.lock().await.remove(delegation_id);
        }
        result
    }

    async fn submit_input(&self, message: &str) -> Result<()> {
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
                            let _ = function_calls;
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
