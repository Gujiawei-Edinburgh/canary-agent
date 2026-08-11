use crate::error::Result;
use crate::program::{EvalProgram, TransitionId};
use crate::vm::{EvalEvent, EvalProjection, EvidenceRef, TransitionDelivery};
use lite_agent_runtime::{Agent, TurnModelEvent, TurnOutcome, TurnStateEvent, TurnStreamEvent};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type RoleFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentObservationStatus {
    Completed,
    Suspended,
    Failed,
    Aborted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentObservationEvent {
    AssistantText {
        text: String,
    },
    FunctionCallsRequested {
        calls: Value,
    },
    FunctionStarted {
        call_id: String,
        name: String,
    },
    FunctionCompleted {
        call_id: String,
        name: String,
    },
    FunctionFailed {
        call_id: String,
        name: String,
        error: String,
    },
    Runtime {
        source: String,
        message: String,
        metadata: Value,
    },
    TokenUsage {
        usage: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentObservation {
    pub status: AgentObservationStatus,
    pub assistant_text: String,
    pub events: Vec<AgentObservationEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentInput {
    pub thread_id: String,
    pub user_text: String,
}

pub type AgentRoleOutput = AgentObservation;

pub trait AgentRoleIo: Send + Sync {
    fn execute<'a>(&'a self, input: AgentInput) -> RoleFuture<'a, AgentObservation>;
}

pub trait TestedAgentIo: AgentRoleIo {}

impl<T> TestedAgentIo for T where T: AgentRoleIo + ?Sized {}

/// Adapts one runtime `Agent` instance to the evaluation I/O contract.
///
/// Hosts can clone this adapter for the tested agent, simulated-user model,
/// and referee model. Each role should use a distinct thread ID so their
/// conversations remain isolated while sharing the same agent instance.
#[derive(Clone)]
pub struct RuntimeAgentIo {
    agent: Arc<Agent>,
}

impl RuntimeAgentIo {
    pub fn new(agent: Arc<Agent>) -> Self {
        Self { agent }
    }

    pub fn agent(&self) -> Arc<Agent> {
        self.agent.clone()
    }

    pub async fn run(&self, input: AgentInput) -> Result<AgentObservation> {
        self.execute(input).await
    }
}

impl AgentRoleIo for RuntimeAgentIo {
    fn execute<'a>(&'a self, input: AgentInput) -> RoleFuture<'a, AgentObservation> {
        Box::pin(async move {
            let AgentInput {
                thread_id,
                user_text,
            } = input;
            let mut assistant_text = String::new();
            let mut events = Vec::new();
            let outcome = self
                .agent
                .run_turn(&thread_id, user_text, Value::Null, |event| match event {
                    TurnStreamEvent::Model(TurnModelEvent::AssistantMessage { text }) => {
                        assistant_text = text.clone();
                        events.push(AgentObservationEvent::AssistantText { text });
                    }
                    TurnStreamEvent::Model(TurnModelEvent::AssistantDelta { .. }) => {}
                    TurnStreamEvent::State(TurnStateEvent::FunctionCallsRequested { calls }) => {
                        events.push(AgentObservationEvent::FunctionCallsRequested {
                            calls: serde_json::to_value(calls).unwrap_or(Value::Null),
                        });
                    }
                    TurnStreamEvent::State(TurnStateEvent::FunctionStarted { call_id, name }) => {
                        events.push(AgentObservationEvent::FunctionStarted { call_id, name })
                    }
                    TurnStreamEvent::State(TurnStateEvent::FunctionCompleted { call_id, name }) => {
                        events.push(AgentObservationEvent::FunctionCompleted { call_id, name })
                    }
                    TurnStreamEvent::State(TurnStateEvent::FunctionFailed {
                        call_id,
                        name,
                        error,
                    }) => events.push(AgentObservationEvent::FunctionFailed {
                        call_id,
                        name,
                        error,
                    }),
                    TurnStreamEvent::State(TurnStateEvent::TurnTokenUsage { usage }) => {
                        events.push(AgentObservationEvent::TokenUsage {
                            usage: serde_json::to_value(usage).unwrap_or(Value::Null),
                        });
                    }
                    TurnStreamEvent::Runtime(runtime) => {
                        events.push(AgentObservationEvent::Runtime {
                            source: runtime.source,
                            message: runtime.message,
                            metadata: runtime.metadata,
                        });
                    }
                    TurnStreamEvent::State(_) | TurnStreamEvent::Model(_) => {}
                })
                .await?;
            let status = match outcome {
                TurnOutcome::AssistantMessage { text } => {
                    if assistant_text.is_empty() {
                        assistant_text = text;
                    }
                    AgentObservationStatus::Completed
                }
                TurnOutcome::Suspended { .. } => AgentObservationStatus::Suspended,
                TurnOutcome::Failed { .. } => AgentObservationStatus::Failed,
                TurnOutcome::Aborted { .. } => AgentObservationStatus::Aborted,
            };
            Ok(AgentObservation {
                status,
                assistant_text,
                events,
            })
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProcessorInput {
    pub program: EvalProgram,
    pub projection: EvalProjection,
    pub latest_observation: Option<AgentObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SimulatedUserCommand {
    SendUserMessage {
        transition: Option<TransitionId>,
        message: String,
    },
    Retry {
        message: String,
        reason: String,
    },
    Commit {
        transition: TransitionId,
        delivery: TransitionDelivery,
        evidence: Vec<EvidenceRef>,
        reason: String,
    },
    Halt {
        reason: String,
    },
}

pub trait SimulatedUserProcessor: Send + Sync {
    fn decide<'a>(&'a self, input: ProcessorInput) -> RoleFuture<'a, SimulatedUserCommand>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RefereeInput {
    pub program: EvalProgram,
    pub projection: EvalProjection,
    pub events: Vec<EvalEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricResult {
    pub name: String,
    pub score: f64,
    pub passed: Option<bool>,
    pub details: Value,
}

pub trait EvalMetric: Send + Sync {
    fn evaluate(&self, input: &RefereeInput) -> MetricResult;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalReport {
    pub metrics: Vec<MetricResult>,
    pub overall_score: Option<f64>,
    pub details: Value,
}

pub trait Referee: Send + Sync {
    fn evaluate<'a>(&'a self, input: RefereeInput) -> RoleFuture<'a, EvalReport>;
}

#[derive(Default)]
pub struct MetricReferee {
    metrics: Vec<Arc<dyn EvalMetric>>,
}

impl MetricReferee {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_metric<M>(mut self, metric: M) -> Self
    where
        M: EvalMetric + 'static,
    {
        self.metrics.push(Arc::new(metric));
        self
    }
}

impl Referee for MetricReferee {
    fn evaluate<'a>(&'a self, input: RefereeInput) -> RoleFuture<'a, EvalReport> {
        Box::pin(async move {
            let metrics = self
                .metrics
                .iter()
                .map(|metric| metric.evaluate(&input))
                .collect::<Vec<_>>();
            let overall_score = (!metrics.is_empty()).then(|| {
                metrics.iter().map(|metric| metric.score).sum::<f64>() / metrics.len() as f64
            });
            Ok(EvalReport {
                metrics,
                overall_score,
                details: json!({}),
            })
        })
    }
}
