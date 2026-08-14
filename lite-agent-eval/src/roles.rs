use crate::environment::{EnvironmentEvent, EnvironmentObservation, EnvironmentSnapshot};
use crate::error::Result;
use lite_agent_runtime::{Agent, TurnModelEvent, TurnOutcome, TurnStateEvent, TurnStreamEvent};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type ActionFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;
pub type EvalReportFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentActionStatus {
    Completed,
    Suspended,
    Failed,
    Aborted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentActionEvent {
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
pub struct AgentAction {
    pub status: AgentActionStatus,
    pub assistant_text: String,
    pub events: Vec<AgentActionEvent>,
}

pub trait EvaluatedPolicy: Send + Sync {
    fn act<'a>(&'a self, observation: EnvironmentObservation) -> ActionFuture<'a, AgentAction>;
}

/// Adapts a runtime `Agent` to the evaluated-policy contract.
#[derive(Clone)]
pub struct RuntimeAgentPolicy {
    agent: Arc<Agent>,
    thread_id: String,
}

impl RuntimeAgentPolicy {
    pub fn new(agent: Arc<Agent>, thread_id: impl Into<String>) -> Self {
        Self {
            agent,
            thread_id: thread_id.into(),
        }
    }

    pub fn agent(&self) -> Arc<Agent> {
        self.agent.clone()
    }
}

impl EvaluatedPolicy for RuntimeAgentPolicy {
    fn act(&self, observation: EnvironmentObservation) -> ActionFuture<'_, AgentAction> {
        Box::pin(async move {
            let mut assistant_text = String::new();
            let mut events = Vec::new();
            let policy_input = render_policy_input(&observation)?;
            let metadata = serde_json::json!({
                "environment": observation.metadata,
                "visible_state": observation.visible_state,
            });
            let outcome = self
                .agent
                .run_turn(
                    &self.thread_id,
                    policy_input,
                    metadata,
                    |event| match event {
                        TurnStreamEvent::Model(TurnModelEvent::AssistantMessage { text }) => {
                            assistant_text = text.clone();
                            events.push(AgentActionEvent::AssistantText { text });
                        }
                        TurnStreamEvent::Model(TurnModelEvent::AssistantDelta { .. }) => {}
                        TurnStreamEvent::State(TurnStateEvent::FunctionCallsRequested {
                            calls,
                        }) => {
                            events.push(AgentActionEvent::FunctionCallsRequested {
                                calls: serde_json::to_value(calls).unwrap_or(Value::Null),
                            });
                        }
                        TurnStreamEvent::State(TurnStateEvent::FunctionStarted {
                            call_id,
                            name,
                        }) => events.push(AgentActionEvent::FunctionStarted { call_id, name }),
                        TurnStreamEvent::State(TurnStateEvent::FunctionCompleted {
                            call_id,
                            name,
                        }) => events.push(AgentActionEvent::FunctionCompleted { call_id, name }),
                        TurnStreamEvent::State(TurnStateEvent::FunctionFailed {
                            call_id,
                            name,
                            error,
                        }) => events.push(AgentActionEvent::FunctionFailed {
                            call_id,
                            name,
                            error,
                        }),
                        TurnStreamEvent::State(TurnStateEvent::TurnTokenUsage { usage }) => {
                            events.push(AgentActionEvent::TokenUsage {
                                usage: serde_json::to_value(usage).unwrap_or(Value::Null),
                            });
                        }
                        TurnStreamEvent::Runtime(runtime) => {
                            events.push(AgentActionEvent::Runtime {
                                source: runtime.source,
                                message: runtime.message,
                                metadata: runtime.metadata,
                            });
                        }
                        TurnStreamEvent::State(_) | TurnStreamEvent::Model(_) => {}
                    },
                )
                .await?;
            let status = match outcome {
                TurnOutcome::AssistantMessage { text } => {
                    if assistant_text.is_empty() {
                        assistant_text = text;
                    }
                    AgentActionStatus::Completed
                }
                TurnOutcome::Suspended { .. } => AgentActionStatus::Suspended,
                TurnOutcome::Failed { .. } => AgentActionStatus::Failed,
                TurnOutcome::Aborted { .. } => AgentActionStatus::Aborted,
            };
            Ok(AgentAction {
                status,
                assistant_text,
                events,
            })
        })
    }
}

fn render_policy_input(observation: &EnvironmentObservation) -> Result<String> {
    let has_visible_state = match &observation.visible_state {
        Value::Null => false,
        Value::Object(values) => !values.is_empty(),
        _ => true,
    };
    if !has_visible_state {
        return Ok(observation.user_text.clone());
    }
    Ok(format!(
        "{}\n\nActive exposed constraints (authoritative):\n{}",
        observation.user_text,
        serde_json::to_string_pretty(&observation.visible_state)?
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RefereeInput {
    pub snapshot: EnvironmentSnapshot,
    pub trajectory: Vec<EnvironmentEvent>,
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
    fn evaluate<'a>(&'a self, input: RefereeInput) -> EvalReportFuture<'a, EvalReport>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn policy_input_includes_the_active_exposed_projection() {
        let rendered = render_policy_input(&EnvironmentObservation {
            user_text: "Continue the task.".to_string(),
            visible_state: json!({"budget": 1000}),
            metadata: Value::Null,
        })
        .expect("policy input");

        assert!(rendered.contains("Continue the task."));
        assert!(rendered.contains("Active exposed constraints"));
        assert!(rendered.contains("\"budget\": 1000"));
    }
}
