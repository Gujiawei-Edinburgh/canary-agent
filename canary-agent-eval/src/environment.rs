use crate::error::{EvalError, Result};
use crate::program::{
    ConstraintDelta, ConstraintId, ConstraintOperation, NodeId, TaskGraph, TransitionId,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExposureOrigin {
    ExplicitDisclosure,
    EnvironmentDerived {
        inputs: Vec<ConstraintId>,
        rule: String,
    },
    ContextProvided {
        source: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExposureRecord {
    pub constraint: ConstraintId,
    pub origin: ExposureOrigin,
    pub observation_sequence: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExposureLedger {
    pub records: BTreeMap<ConstraintId, ExposureRecord>,
}

impl ExposureLedger {
    pub fn contains(&self, constraint: &ConstraintId) -> bool {
        self.records.contains_key(constraint)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentState {
    pub status: EnvironmentStatus,
    pub current_node: NodeId,
    pub constraints: ConstraintLedger,
    pub exposures: ExposureLedger,
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
    pub exposures: Vec<ConstraintExposure>,
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
    fn realize(&self, input: ObservationRealizerInput)
    -> EnvironmentFuture<'_, ObservationContent>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConstraintExposure {
    Disclose {
        constraint: ConstraintId,
    },
    Derive {
        constraint: ConstraintId,
        inputs: Vec<ConstraintId>,
        rule: String,
    },
    ProvideContext {
        constraint: ConstraintId,
        source: String,
    },
}

impl ConstraintExposure {
    fn constraint(&self) -> &ConstraintId {
        match self {
            Self::Disclose { constraint }
            | Self::Derive { constraint, .. }
            | Self::ProvideContext { constraint, .. } => constraint,
        }
    }

    fn origin(&self) -> ExposureOrigin {
        match self {
            Self::Disclose { .. } => ExposureOrigin::ExplicitDisclosure,
            Self::Derive { inputs, rule, .. } => ExposureOrigin::EnvironmentDerived {
                inputs: inputs.clone(),
                rule: rule.clone(),
            },
            Self::ProvideContext { source, .. } => ExposureOrigin::ContextProvided {
                source: source.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EnvironmentDecision {
    Transition {
        transition: TransitionId,
        evidence: Vec<EvidenceRef>,
        reason: String,
    },
    Retry {
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
        exposures: Vec<ExposureRecord>,
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
    fn reset(&mut self) -> EnvironmentFuture<'_, EnvironmentOutput>;

    fn step(&mut self, action: AgentAction) -> EnvironmentFuture<'_, EnvironmentOutput>;

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
                exposures: ExposureLedger::default(),
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
            EnvironmentDecision::Retry { reason } => Ok(ObservationCause::Retry {
                reason: reason.clone(),
            }),
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
            ConstraintOperation::Add { id, value } => {
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
            }
            ConstraintOperation::Replace { target, id, value } => {
                if self.state.constraints.active.remove(target).is_none() {
                    return Err(EvalError::InvalidEnvironmentAction(format!(
                        "cannot replace inactive constraint: {}",
                        target.0
                    )));
                }
                self.state
                    .constraints
                    .active
                    .insert(id.clone(), value.clone());
            }
            ConstraintOperation::Remove { target } => {
                if self.state.constraints.active.remove(target).is_none() {
                    return Err(EvalError::InvalidEnvironmentAction(format!(
                        "cannot remove inactive constraint: {}",
                        target.0
                    )));
                }
            }
        }
        self.state.constraints.applied.push(ConstraintApplication {
            node: node.clone(),
            delta,
        });
        Ok(())
    }

    fn prepare_exposure_records(
        &self,
        exposures: &[ConstraintExposure],
        observation_sequence: u64,
    ) -> Result<Vec<ExposureRecord>> {
        let mut available = self
            .state
            .exposures
            .records
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut records = Vec::with_capacity(exposures.len());

        for exposure in exposures {
            let constraint = exposure.constraint();
            if !self.state.constraints.active.contains_key(constraint) {
                return Err(EvalError::InvalidEnvironmentAction(format!(
                    "exposure references inactive constraint: {}",
                    constraint.0
                )));
            }
            if available.contains(constraint) {
                return Err(EvalError::InvalidEnvironmentAction(format!(
                    "constraint is already exposed: {}",
                    constraint.0
                )));
            }

            match exposure {
                ConstraintExposure::Disclose { .. } => {}
                ConstraintExposure::Derive { inputs, rule, .. } => {
                    if inputs.is_empty() {
                        return Err(EvalError::InvalidEnvironmentAction(format!(
                            "derived exposure has no inputs: {}",
                            constraint.0
                        )));
                    }
                    if rule.trim().is_empty() {
                        return Err(EvalError::InvalidEnvironmentAction(format!(
                            "derived exposure has an empty rule: {}",
                            constraint.0
                        )));
                    }
                    for input in inputs {
                        if !self.state.constraints.active.contains_key(input) {
                            return Err(EvalError::InvalidEnvironmentAction(format!(
                                "derived exposure references inactive input {} for {}",
                                input.0, constraint.0
                            )));
                        }
                        if !available.contains(input) {
                            return Err(EvalError::InvalidEnvironmentAction(format!(
                                "derived exposure references unexposed input {} for {}",
                                input.0, constraint.0
                            )));
                        }
                    }
                }
                ConstraintExposure::ProvideContext { source, .. } => {
                    if source.trim().is_empty() {
                        return Err(EvalError::InvalidEnvironmentAction(format!(
                            "context-provided exposure has an empty source: {}",
                            constraint.0
                        )));
                    }
                }
            }

            available.insert(constraint.clone());
            records.push(ExposureRecord {
                constraint: constraint.clone(),
                origin: exposure.origin(),
                observation_sequence,
            });
        }

        Ok(records)
    }

    fn visible_state(&self) -> Value {
        let mut visible = serde_json::Map::new();
        for id in self.state.exposures.records.keys() {
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
        let observation_sequence = self.state.step.saturating_add(1);
        let exposure_records =
            self.prepare_exposure_records(&content.exposures, observation_sequence)?;
        for record in &exposure_records {
            self.state
                .exposures
                .records
                .insert(record.constraint.clone(), record.clone());
        }
        let observation = EnvironmentObservation {
            user_text: content.user_text,
            visible_state: self.visible_state(),
            metadata: content.metadata,
        };
        self.record(EnvironmentEventKind::ObservationEmitted {
            observation: observation.clone(),
            exposures: exposure_records,
        });
        Ok(observation)
    }
}

impl EvalEnvironment for GraphEnvironment {
    fn reset(&mut self) -> EnvironmentFuture<'_, EnvironmentOutput> {
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

    fn step(&mut self, action: AgentAction) -> EnvironmentFuture<'_, EnvironmentOutput> {
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
        fn decide(
            &self,
            _input: EnvironmentControllerInput,
        ) -> EnvironmentFuture<'_, EnvironmentDecision> {
            Box::pin(async {
                Ok(EnvironmentDecision::Transition {
                    transition: TransitionId::from("finish"),
                    evidence: Vec::new(),
                    reason: "completed".to_string(),
                })
            })
        }
    }

    struct StaticRealizer;

    impl ObservationRealizer for StaticRealizer {
        fn realize(
            &self,
            input: ObservationRealizerInput,
        ) -> EnvironmentFuture<'_, ObservationContent> {
            Box::pin(async move {
                Ok(ObservationContent {
                    user_text: match &input.cause {
                        ObservationCause::Reset => "start".to_string(),
                        _ => "continue".to_string(),
                    },
                    exposures: match &input.cause {
                        ObservationCause::Reset => vec![
                            ConstraintExposure::Disclose {
                                constraint: ConstraintId::from("premise"),
                            },
                            ConstraintExposure::Derive {
                                constraint: ConstraintId::from("computed"),
                                inputs: vec![ConstraintId::from("premise")],
                                rule: "copy premise".to_string(),
                            },
                            ConstraintExposure::ProvideContext {
                                constraint: ConstraintId::from("context"),
                                source: "evaluation fixture".to_string(),
                            },
                        ],
                        _ => Vec::new(),
                    },
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
                                id: ConstraintId::from("premise"),
                                value: json!("known premise"),
                            },
                            provenance: None,
                        },
                        ConstraintDelta {
                            operation: ConstraintOperation::Add {
                                id: ConstraintId::from("computed"),
                                value: json!("derived value"),
                            },
                            provenance: None,
                        },
                        ConstraintDelta {
                            operation: ConstraintOperation::Add {
                                id: ConstraintId::from("context"),
                                value: json!(true),
                            },
                            provenance: None,
                        },
                        ConstraintDelta {
                            operation: ConstraintOperation::Add {
                                id: ConstraintId::from("unexposed"),
                                value: json!("hidden"),
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
    async fn observation_atomically_records_exposures_and_filters_unexposed_state() {
        let mut environment = GraphEnvironment::new(graph(), CompleteController, StaticRealizer)
            .expect("environment");

        let initial = environment.reset().await.expect("reset");
        assert_eq!(initial.status, EnvironmentStatus::Running);
        assert_eq!(
            initial.observation.expect("observation").visible_state,
            json!({
                "context": true,
                "computed": "derived value",
                "premise": "known premise"
            })
        );

        let observation_event = environment
            .events
            .iter()
            .find_map(|event| match &event.kind {
                EnvironmentEventKind::ObservationEmitted { exposures, .. } => Some(exposures),
                _ => None,
            })
            .expect("observation event");
        assert_eq!(observation_event.len(), 3);
        assert!(
            observation_event
                .iter()
                .all(|record| record.observation_sequence == 2)
        );
        assert!(
            !environment
                .state()
                .exposures
                .contains(&ConstraintId::from("unexposed"))
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
                .exposures
                .contains(&ConstraintId::from("premise"))
        );
        assert!(matches!(
            environment.state().termination,
            Some(TerminationReason::TerminalNode { .. })
        ));
    }

    struct InvalidRealizer;

    impl ObservationRealizer for InvalidRealizer {
        fn realize(
            &self,
            _input: ObservationRealizerInput,
        ) -> EnvironmentFuture<'_, ObservationContent> {
            Box::pin(async {
                Ok(ObservationContent {
                    user_text: "invalid duplicate exposure".to_string(),
                    exposures: vec![
                        ConstraintExposure::Disclose {
                            constraint: ConstraintId::from("premise"),
                        },
                        ConstraintExposure::ProvideContext {
                            constraint: ConstraintId::from("premise"),
                            source: "duplicate".to_string(),
                        },
                    ],
                    metadata: Value::Null,
                })
            })
        }
    }

    #[tokio::test]
    async fn invalid_exposure_does_not_partially_update_the_ledger() {
        let mut environment = GraphEnvironment::new(graph(), CompleteController, InvalidRealizer)
            .expect("environment");

        let error = environment.reset().await.expect_err("duplicate exposure");
        assert!(error.to_string().contains("already exposed"));
        assert!(environment.state().exposures.records.is_empty());
        assert!(
            !environment
                .events
                .iter()
                .any(|event| matches!(event.kind, EnvironmentEventKind::ObservationEmitted { .. }))
        );
    }

    struct RevisionController;

    impl EnvironmentController for RevisionController {
        fn decide(
            &self,
            _input: EnvironmentControllerInput,
        ) -> EnvironmentFuture<'_, EnvironmentDecision> {
            Box::pin(async {
                Ok(EnvironmentDecision::Transition {
                    transition: TransitionId::from("revise"),
                    evidence: Vec::new(),
                    reason: "requirement changed".to_string(),
                })
            })
        }
    }

    struct RevisionRealizer;

    impl ObservationRealizer for RevisionRealizer {
        fn realize(
            &self,
            input: ObservationRealizerInput,
        ) -> EnvironmentFuture<'_, ObservationContent> {
            Box::pin(async move {
                Ok(ObservationContent {
                    user_text: "requirement".to_string(),
                    exposures: match input.cause {
                        ObservationCause::Reset => vec![ConstraintExposure::Disclose {
                            constraint: ConstraintId::from("old"),
                        }],
                        _ => Vec::new(),
                    },
                    metadata: Value::Null,
                })
            })
        }
    }

    fn revision_graph() -> TaskGraph {
        TaskCase {
            id: "revision".to_string(),
            version: "1".to_string(),
            start: NodeId::from("start"),
            nodes: vec![
                TaskNode {
                    id: NodeId::from("start"),
                    constraints: vec![ConstraintDelta {
                        operation: ConstraintOperation::Add {
                            id: ConstraintId::from("old"),
                            value: json!("old value"),
                        },
                        provenance: None,
                    }],
                    result_obligation: None,
                    evidence_obligation: None,
                    terminal: false,
                },
                TaskNode {
                    id: NodeId::from("revised"),
                    constraints: vec![ConstraintDelta {
                        operation: ConstraintOperation::Replace {
                            target: ConstraintId::from("old"),
                            id: ConstraintId::from("new"),
                            value: json!("new value"),
                        },
                        provenance: None,
                    }],
                    result_obligation: None,
                    evidence_obligation: None,
                    terminal: false,
                },
            ],
            transitions: vec![TaskTransition {
                id: TransitionId::from("revise"),
                from: NodeId::from("start"),
                to: NodeId::from("revised"),
                kind: TransitionKind::Revision,
            }],
        }
        .compile()
        .expect("revision graph")
    }

    #[tokio::test]
    async fn replacement_does_not_inherit_exposure() {
        let mut environment =
            GraphEnvironment::new(revision_graph(), RevisionController, RevisionRealizer)
                .expect("environment");
        environment.reset().await.expect("reset");

        let output = environment
            .step(AgentAction {
                status: AgentActionStatus::Completed,
                assistant_text: "done".to_string(),
                events: Vec::new(),
            })
            .await
            .expect("revision");

        assert_eq!(
            output.observation.expect("observation").visible_state,
            json!({})
        );
        assert!(
            environment
                .state()
                .exposures
                .contains(&ConstraintId::from("old"))
        );
        assert!(
            !environment
                .state()
                .exposures
                .contains(&ConstraintId::from("new"))
        );
    }
}
