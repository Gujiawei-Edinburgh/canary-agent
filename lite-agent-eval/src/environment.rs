use crate::error::{EvalError, Result};
use crate::program::{
    ActivationPolicy, ConstraintDelta, ConstraintId, ConstraintOperation, NodeId, TaskGraph,
    TransitionId,
};
use crate::roles::AgentAction;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type EnvironmentFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceRef {
    pub kind: String,
    pub reference: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentStatus {
    NotStarted,
    Running,
    Terminated,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TerminationReason {
    TerminalNode { node: NodeId },
    Controller { reason: String },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ConstraintLedger {
    pub active: BTreeMap<ConstraintId, Value>,
    pub applied: Vec<ConstraintApplication>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConstraintApplication {
    pub node: NodeId,
    pub delta: ConstraintDelta,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct VisibilityState {
    pub disclosed: BTreeSet<ConstraintId>,
    pub derived: BTreeSet<ConstraintId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentState {
    pub status: EnvironmentStatus,
    pub current_node: NodeId,
    pub constraints: ConstraintLedger,
    pub visibility: VisibilityState,
    pub termination: Option<TerminationReason>,
    pub step: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentSnapshot {
    pub environment_id: String,
    pub version: String,
    pub status: EnvironmentStatus,
    pub step: u64,
    pub state: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentObservation {
    pub user_text: String,
    pub visible_state: Value,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObservationContent {
    pub user_text: String,
    pub visibility: Vec<VisibilityChange>,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ObservationCause {
    Reset,
    Transition { transition: TransitionId },
    Retry { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObservationRealizerInput {
    pub graph: TaskGraph,
    pub state: EnvironmentState,
    pub cause: ObservationCause,
    pub latest_action: Option<AgentAction>,
}

pub trait ObservationRealizer: Send + Sync {
    fn realize<'a>(
        &'a self,
        input: ObservationRealizerInput,
    ) -> EnvironmentFuture<'a, ObservationContent>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VisibilityChange {
    Disclose { constraint: ConstraintId },
    Derive { constraint: ConstraintId },
    Conceal { constraint: ConstraintId },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EnvironmentDecision {
    Transition {
        transition: TransitionId,
        visibility: Vec<VisibilityChange>,
        evidence: Vec<EvidenceRef>,
        reason: String,
    },
    Retry {
        visibility: Vec<VisibilityChange>,
        reason: String,
    },
    Terminate {
        reason: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentControllerInput {
    pub graph: TaskGraph,
    pub state: EnvironmentState,
    pub action: AgentAction,
}

pub trait EnvironmentController: Send + Sync {
    fn decide<'a>(
        &'a self,
        input: EnvironmentControllerInput,
    ) -> EnvironmentFuture<'a, EnvironmentDecision>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentOutput {
    pub observation: Option<EnvironmentObservation>,
    pub status: EnvironmentStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentEvent {
    pub sequence: u64,
    pub kind: EnvironmentEventKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EnvironmentEventKind {
    Started {
        node: NodeId,
    },
    ObservationEmitted {
        observation: EnvironmentObservation,
    },
    AgentActionRecorded {
        action: AgentAction,
    },
    DecisionMade {
        decision: EnvironmentDecision,
    },
    TransitionApplied {
        transition: TransitionId,
        from: NodeId,
        to: NodeId,
        evidence: Vec<EvidenceRef>,
    },
    Terminated {
        reason: TerminationReason,
    },
    Custom {
        kind: String,
        payload: Value,
    },
}

pub trait EvalEnvironment: Send {
    fn reset<'a>(&'a mut self) -> EnvironmentFuture<'a, EnvironmentOutput>;

    fn step<'a>(&'a mut self, action: AgentAction) -> EnvironmentFuture<'a, EnvironmentOutput>;

    fn snapshot(&self) -> Result<EnvironmentSnapshot>;

    fn trajectory(&self) -> Vec<EnvironmentEvent>;
}

pub struct GraphEnvironment {
    graph: TaskGraph,
    state: EnvironmentState,
    controller: Arc<dyn EnvironmentController>,
    realizer: Arc<dyn ObservationRealizer>,
    events: Vec<EnvironmentEvent>,
}

impl GraphEnvironment {
    pub fn new<C, R>(graph: TaskGraph, controller: C, realizer: R) -> Result<Self>
    where
        C: EnvironmentController + 'static,
        R: ObservationRealizer + 'static,
    {
        let start = graph.start.clone();
        let mut environment = Self {
            graph,
            state: EnvironmentState {
                status: EnvironmentStatus::NotStarted,
                current_node: start.clone(),
                constraints: ConstraintLedger::default(),
                visibility: VisibilityState::default(),
                termination: None,
                step: 0,
            },
            controller: Arc::new(controller),
            realizer: Arc::new(realizer),
            events: Vec::new(),
        };
        environment.apply_node(&start)?;
        Ok(environment)
    }

    pub fn graph(&self) -> &TaskGraph {
        &self.graph
    }

    pub fn state(&self) -> &EnvironmentState {
        &self.state
    }

    fn apply_decision(&mut self, decision: &EnvironmentDecision) -> Result<ObservationCause> {
        match decision {
            EnvironmentDecision::Transition {
                transition,
                visibility,
                evidence,
                ..
            } => {
                let transition_spec = self.graph.transition(transition)?.clone();
                if transition_spec.from != self.state.current_node {
                    return Err(EvalError::InvalidEnvironmentAction(format!(
                        "transition {} does not start at current node",
                        transition.0
                    )));
                }

                let previous_state = self.state.clone();
                if let Err(error) = self.apply_node(&transition_spec.to) {
                    self.state = previous_state;
                    return Err(error);
                }
                if let Err(error) = self.apply_visibility_changes(visibility) {
                    self.state = previous_state;
                    return Err(error);
                }

                let from = self.state.current_node.clone();
                self.state.current_node = transition_spec.to.clone();
                self.record(EnvironmentEventKind::TransitionApplied {
                    transition: transition.clone(),
                    from,
                    to: transition_spec.to.clone(),
                    evidence: evidence.clone(),
                });
                if self.graph.node(&transition_spec.to)?.terminal {
                    self.terminate(TerminationReason::TerminalNode {
                        node: transition_spec.to.clone(),
                    });
                }
                Ok(ObservationCause::Transition {
                    transition: transition.clone(),
                })
            }
            EnvironmentDecision::Retry { visibility, reason } => {
                self.apply_visibility_changes(visibility)?;
                Ok(ObservationCause::Retry {
                    reason: reason.clone(),
                })
            }
            EnvironmentDecision::Terminate { reason } => {
                self.terminate(TerminationReason::Controller {
                    reason: reason.clone(),
                });
                Ok(ObservationCause::Retry {
                    reason: reason.clone(),
                })
            }
        }
    }

    fn apply_node(&mut self, node_id: &NodeId) -> Result<()> {
        let node = self.graph.node(node_id)?.clone();
        let previous_state = self.state.clone();
        for delta in node.constraints {
            if let Err(error) = self.apply_constraint(node_id, delta) {
                self.state = previous_state;
                return Err(error);
            }
        }
        Ok(())
    }

    fn apply_constraint(&mut self, node: &NodeId, delta: ConstraintDelta) -> Result<()> {
        match &delta.operation {
            ConstraintOperation::Add {
                id,
                value,
                activation,
            } => {
                if self
                    .state
                    .constraints
                    .active
                    .insert(id.clone(), value.clone())
                    .is_some()
                {
                    return Err(EvalError::InvalidEnvironmentAction(format!(
                        "constraint already active: {}",
                        id.0
                    )));
                }
                match activation {
                    ActivationPolicy::ExplicitDisclosure => {}
                    ActivationPolicy::AlreadyAuthorized => {
                        self.state.visibility.disclosed.insert(id.clone());
                    }
                    ActivationPolicy::Derivable => {
                        self.state.visibility.derived.insert(id.clone());
                    }
                }
            }
            ConstraintOperation::Replace { target, id, value } => {
                if self.state.constraints.active.remove(target).is_none() {
                    return Err(EvalError::InvalidEnvironmentAction(format!(
                        "cannot replace inactive constraint: {}",
                        target.0
                    )));
                }
                let was_disclosed = self.state.visibility.disclosed.remove(target);
                let was_derived = self.state.visibility.derived.remove(target);
                self.state
                    .constraints
                    .active
                    .insert(id.clone(), value.clone());
                if was_disclosed {
                    self.state.visibility.disclosed.insert(id.clone());
                }
                if was_derived {
                    self.state.visibility.derived.insert(id.clone());
                }
            }
            ConstraintOperation::Remove { target } => {
                if self.state.constraints.active.remove(target).is_none() {
                    return Err(EvalError::InvalidEnvironmentAction(format!(
                        "cannot remove inactive constraint: {}",
                        target.0
                    )));
                }
                self.state.visibility.disclosed.remove(target);
                self.state.visibility.derived.remove(target);
            }
        }
        self.state.constraints.applied.push(ConstraintApplication {
            node: node.clone(),
            delta,
        });
        Ok(())
    }

    fn apply_visibility_changes(&mut self, changes: &[VisibilityChange]) -> Result<()> {
        for change in changes {
            let id = match change {
                VisibilityChange::Disclose { constraint }
                | VisibilityChange::Derive { constraint }
                | VisibilityChange::Conceal { constraint } => constraint,
            };
            if !self.state.constraints.active.contains_key(id) {
                return Err(EvalError::InvalidEnvironmentAction(format!(
                    "visibility change references inactive constraint: {}",
                    id.0
                )));
            }
            match change {
                VisibilityChange::Disclose { constraint } => {
                    self.state.visibility.derived.remove(constraint);
                    self.state.visibility.disclosed.insert(constraint.clone());
                }
                VisibilityChange::Derive { constraint } => {
                    self.state.visibility.disclosed.remove(constraint);
                    self.state.visibility.derived.insert(constraint.clone());
                }
                VisibilityChange::Conceal { constraint } => {
                    self.state.visibility.disclosed.remove(constraint);
                    self.state.visibility.derived.remove(constraint);
                }
            }
        }
        Ok(())
    }

    fn visible_state(&self) -> Value {
        let mut visible = serde_json::Map::new();
        for id in self
            .state
            .visibility
            .disclosed
            .iter()
            .chain(self.state.visibility.derived.iter())
        {
            if let Some(value) = self.state.constraints.active.get(id) {
                visible.insert(id.0.clone(), value.clone());
            }
        }
        Value::Object(visible)
    }

    fn terminate(&mut self, reason: TerminationReason) {
        self.state.status = EnvironmentStatus::Terminated;
        self.state.termination = Some(reason.clone());
        self.record(EnvironmentEventKind::Terminated { reason });
    }

    fn record(&mut self, kind: EnvironmentEventKind) {
        self.state.step = self.state.step.saturating_add(1);
        self.events.push(EnvironmentEvent {
            sequence: self.state.step,
            kind,
        });
    }

    async fn emit_observation(
        &mut self,
        cause: ObservationCause,
        latest_action: Option<AgentAction>,
    ) -> Result<EnvironmentObservation> {
        let content = self
            .realizer
            .realize(ObservationRealizerInput {
                graph: self.graph.clone(),
                state: self.state.clone(),
                cause,
                latest_action,
            })
            .await?;
        if content.user_text.trim().is_empty() {
            return Err(EvalError::InvalidEnvironmentAction(
                "observation text is empty".to_string(),
            ));
        }
        self.apply_visibility_changes(&content.visibility)?;
        let observation = EnvironmentObservation {
            user_text: content.user_text,
            visible_state: self.visible_state(),
            metadata: content.metadata,
        };
        self.record(EnvironmentEventKind::ObservationEmitted {
            observation: observation.clone(),
        });
        Ok(observation)
    }
}

impl EvalEnvironment for GraphEnvironment {
    fn reset<'a>(&'a mut self) -> EnvironmentFuture<'a, EnvironmentOutput> {
        Box::pin(async move {
            if self.state.status != EnvironmentStatus::NotStarted {
                return Err(EvalError::InvalidEnvironmentAction(
                    "environment has already been reset".to_string(),
                ));
            }
            self.state.status = EnvironmentStatus::Running;
            self.record(EnvironmentEventKind::Started {
                node: self.state.current_node.clone(),
            });
            if self.graph.node(&self.state.current_node)?.terminal {
                self.terminate(TerminationReason::TerminalNode {
                    node: self.state.current_node.clone(),
                });
                return Ok(EnvironmentOutput {
                    observation: None,
                    status: self.state.status,
                });
            }
            let observation = self.emit_observation(ObservationCause::Reset, None).await?;
            Ok(EnvironmentOutput {
                observation: Some(observation),
                status: self.state.status,
            })
        })
    }

    fn step<'a>(&'a mut self, action: AgentAction) -> EnvironmentFuture<'a, EnvironmentOutput> {
        Box::pin(async move {
            if self.state.status != EnvironmentStatus::Running {
                return Err(EvalError::InvalidEnvironmentAction(
                    "environment is not running".to_string(),
                ));
            }
            self.record(EnvironmentEventKind::AgentActionRecorded {
                action: action.clone(),
            });
            let decision = self
                .controller
                .decide(EnvironmentControllerInput {
                    graph: self.graph.clone(),
                    state: self.state.clone(),
                    action: action.clone(),
                })
                .await?;
            self.record(EnvironmentEventKind::DecisionMade {
                decision: decision.clone(),
            });
            let cause = self.apply_decision(&decision)?;
            if self.state.status == EnvironmentStatus::Terminated {
                return Ok(EnvironmentOutput {
                    observation: None,
                    status: self.state.status,
                });
            }
            let observation = self.emit_observation(cause, Some(action)).await?;
            Ok(EnvironmentOutput {
                observation: Some(observation),
                status: self.state.status,
            })
        })
    }

    fn snapshot(&self) -> Result<EnvironmentSnapshot> {
        Ok(EnvironmentSnapshot {
            environment_id: self.graph.case_id.clone(),
            version: self.graph.version.clone(),
            status: self.state.status,
            step: self.state.step,
            state: serde_json::to_value(&self.state)?,
        })
    }

    fn trajectory(&self) -> Vec<EnvironmentEvent> {
        self.events.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{TaskCase, TaskNode, TaskTransition, TransitionKind};
    use crate::roles::{AgentActionEvent, AgentActionStatus};
    use serde_json::json;

    struct CompleteController;

    impl EnvironmentController for CompleteController {
        fn decide<'a>(
            &'a self,
            _input: EnvironmentControllerInput,
        ) -> EnvironmentFuture<'a, EnvironmentDecision> {
            Box::pin(async {
                Ok(EnvironmentDecision::Transition {
                    transition: TransitionId::from("finish"),
                    visibility: vec![VisibilityChange::Disclose {
                        constraint: ConstraintId::from("hidden"),
                    }],
                    evidence: Vec::new(),
                    reason: "completed".to_string(),
                })
            })
        }
    }

    struct StaticRealizer;

    impl ObservationRealizer for StaticRealizer {
        fn realize<'a>(
            &'a self,
            input: ObservationRealizerInput,
        ) -> EnvironmentFuture<'a, ObservationContent> {
            Box::pin(async move {
                Ok(ObservationContent {
                    user_text: match input.cause {
                        ObservationCause::Reset => "start".to_string(),
                        _ => "continue".to_string(),
                    },
                    visibility: Vec::new(),
                    metadata: Value::Null,
                })
            })
        }
    }

    fn graph() -> TaskGraph {
        TaskCase {
            id: "environment".to_string(),
            version: "1".to_string(),
            start: NodeId::from("start"),
            nodes: vec![
                TaskNode {
                    id: NodeId::from("start"),
                    constraints: vec![
                        ConstraintDelta {
                            operation: ConstraintOperation::Add {
                                id: ConstraintId::from("hidden"),
                                value: json!("secret"),
                                activation: ActivationPolicy::ExplicitDisclosure,
                            },
                            provenance: None,
                        },
                        ConstraintDelta {
                            operation: ConstraintOperation::Add {
                                id: ConstraintId::from("authorized"),
                                value: json!(true),
                                activation: ActivationPolicy::AlreadyAuthorized,
                            },
                            provenance: None,
                        },
                    ],
                    result_obligation: None,
                    evidence_obligation: None,
                    terminal: false,
                },
                TaskNode {
                    id: NodeId::from("done"),
                    constraints: Vec::new(),
                    result_obligation: None,
                    evidence_obligation: None,
                    terminal: true,
                },
            ],
            transitions: vec![TaskTransition {
                id: TransitionId::from("finish"),
                from: NodeId::from("start"),
                to: NodeId::from("done"),
                kind: TransitionKind::Progress,
            }],
        }
        .compile()
        .expect("task graph")
    }

    #[tokio::test]
    async fn environment_filters_latent_state_and_applies_transition() {
        let mut environment = GraphEnvironment::new(graph(), CompleteController, StaticRealizer)
            .expect("environment");

        let initial = environment.reset().await.expect("reset");
        assert_eq!(initial.status, EnvironmentStatus::Running);
        assert_eq!(
            initial.observation.expect("observation").visible_state,
            json!({"authorized": true})
        );

        let output = environment
            .step(AgentAction {
                status: AgentActionStatus::Completed,
                assistant_text: "done".to_string(),
                events: Vec::<AgentActionEvent>::new(),
            })
            .await
            .expect("step");
        assert_eq!(output.status, EnvironmentStatus::Terminated);
        assert!(output.observation.is_none());
        assert!(
            environment
                .state()
                .visibility
                .disclosed
                .contains(&ConstraintId::from("hidden"))
        );
        assert!(matches!(
            environment.state().termination,
            Some(TerminationReason::TerminalNode { .. })
        ));
    }
}
