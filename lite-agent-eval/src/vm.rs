use crate::error::{EvalError, Result};
use crate::program::{
    ConstraintDelta, ConstraintId, ConstraintOperation, EvalProgram, NodeId, TransitionId,
};
use crate::roles::{
    AgentInput, AgentObservation, ProcessorInput, Referee, RefereeInput, SimulatedUserCommand,
    SimulatedUserProcessor, TestedAgentIo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceRef {
    pub kind: String,
    pub reference: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransitionDelivery {
    Explicit,
    Epsilon,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvalVmStatus {
    Running,
    Completed,
    Halted,
    Failed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvalPhase {
    AwaitingUserAction,
    AwaitingAgentObservation,
    AwaitingSimulatorDecision,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConstraintLedger {
    pub active: BTreeMap<ConstraintId, Value>,
    pub applied: Vec<ConstraintApplication>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConstraintApplication {
    pub node: NodeId,
    pub delta: ConstraintDelta,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalProjection {
    pub status: EvalVmStatus,
    pub phase: EvalPhase,
    pub current_node: NodeId,
    pub pending_transition: Option<TransitionId>,
    pub constraints: ConstraintLedger,
    pub step: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalEvent {
    pub sequence: u64,
    pub kind: EvalEventKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvalEventKind {
    ProgramStarted {
        node: NodeId,
    },
    UserMessage {
        message: String,
    },
    AgentObservation {
        observation: AgentObservation,
    },
    SimulatorCommand {
        command: SimulatedUserCommand,
    },
    TransitionCommitted {
        transition: Option<TransitionId>,
        from: NodeId,
        to: NodeId,
        delivery: TransitionDelivery,
        evidence: Vec<EvidenceRef>,
    },
    Halted {
        reason: String,
    },
}

pub struct EvalVmComponents {
    pub tested_agent: std::sync::Arc<dyn TestedAgentIo>,
    pub simulated_user: std::sync::Arc<dyn SimulatedUserProcessor>,
    pub referee: std::sync::Arc<dyn Referee>,
}

impl EvalVmComponents {
    pub fn new<T, S, R>(tested_agent: T, simulated_user: S, referee: R) -> Self
    where
        T: TestedAgentIo + 'static,
        S: SimulatedUserProcessor + 'static,
        R: Referee + 'static,
    {
        Self {
            tested_agent: std::sync::Arc::new(tested_agent),
            simulated_user: std::sync::Arc::new(simulated_user),
            referee: std::sync::Arc::new(referee),
        }
    }
}

pub struct EvalVm {
    program: EvalProgram,
    projection: EvalProjection,
    events: Vec<EvalEvent>,
    components: Option<EvalVmComponents>,
}

impl EvalVm {
    pub fn new(program: EvalProgram, components: EvalVmComponents) -> Result<Self> {
        Self::build(program, Some(components))
    }

    /// Constructs a VM without role components for reducer-level testing.
    pub fn reducer(program: EvalProgram) -> Result<Self> {
        Self::build(program, None)
    }

    fn build(program: EvalProgram, components: Option<EvalVmComponents>) -> Result<Self> {
        let start = program.start.clone();
        let mut vm = Self {
            program,
            projection: EvalProjection {
                status: EvalVmStatus::Running,
                phase: EvalPhase::AwaitingUserAction,
                current_node: start.clone(),
                pending_transition: None,
                constraints: ConstraintLedger {
                    active: BTreeMap::new(),
                    applied: Vec::new(),
                },
                step: 0,
            },
            events: Vec::new(),
            components,
        };
        vm.apply_node_constraints(&start)?;
        vm.record(EvalEventKind::ProgramStarted { node: start });
        if vm.program.node(&vm.projection.current_node)?.terminal {
            vm.projection.status = EvalVmStatus::Completed;
        }
        Ok(vm)
    }

    /// Runs the VM by coordinating its three owned components.
    pub async fn run(&mut self) -> Result<crate::roles::EvalReport> {
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| EvalError::Role("VM has no execution components".to_string()))?;
        let tested_agent = components.tested_agent.clone();
        let simulated_user = components.simulated_user.clone();
        let referee = components.referee.clone();

        while self.projection.status == EvalVmStatus::Running {
            let command = simulated_user
                .decide(ProcessorInput {
                    program: self.program.clone(),
                    projection: self.projection.clone(),
                    latest_observation: self.latest_observation(),
                })
                .await?;

            let agent_input = match &command {
                SimulatedUserCommand::SendUserMessage { message, .. }
                | SimulatedUserCommand::Retry { message, .. } => Some(message.clone()),
                SimulatedUserCommand::Commit { .. } | SimulatedUserCommand::Halt { .. } => None,
            };
            self.apply_command(command)?;

            if let Some(user_text) = agent_input {
                let observation = tested_agent
                    .execute(AgentInput {
                        thread_id: format!("eval:{}", self.program.case_id),
                        user_text,
                    })
                    .await?;
                self.observe_agent(observation)?;
            }
        }

        referee
            .evaluate(RefereeInput {
                program: self.program.clone(),
                projection: self.projection.clone(),
                events: self.events.clone(),
            })
            .await
    }

    fn latest_observation(&self) -> Option<AgentObservation> {
        self.events
            .iter()
            .rev()
            .find_map(|event| match &event.kind {
                EvalEventKind::AgentObservation { observation } => Some(observation.clone()),
                _ => None,
            })
    }

    pub fn program(&self) -> &EvalProgram {
        &self.program
    }

    pub fn projection(&self) -> &EvalProjection {
        &self.projection
    }

    pub fn events(&self) -> &[EvalEvent] {
        &self.events
    }

    pub fn observe_agent(&mut self, observation: AgentObservation) -> Result<()> {
        self.require_phase(EvalPhase::AwaitingAgentObservation)?;
        self.record(EvalEventKind::AgentObservation { observation });
        self.projection.phase = EvalPhase::AwaitingSimulatorDecision;
        Ok(())
    }

    pub fn apply_command(&mut self, command: SimulatedUserCommand) -> Result<()> {
        self.record(EvalEventKind::SimulatorCommand {
            command: command.clone(),
        });
        match command {
            SimulatedUserCommand::SendUserMessage {
                transition,
                message,
            } => self.send_user_message(transition, message),
            SimulatedUserCommand::Retry { message, .. } => self.retry(message),
            SimulatedUserCommand::Commit {
                transition,
                delivery,
                evidence,
                ..
            } => self.commit(transition, delivery, evidence),
            SimulatedUserCommand::Halt { reason } => {
                if self.projection.status != EvalVmStatus::Running {
                    return Err(EvalError::InvalidCommand(
                        "evaluation is already stopped".to_string(),
                    ));
                }
                self.projection.status = EvalVmStatus::Halted;
                self.projection.phase = EvalPhase::AwaitingSimulatorDecision;
                self.record(EvalEventKind::Halted { reason });
                Ok(())
            }
        }
    }

    fn send_user_message(
        &mut self,
        transition: Option<TransitionId>,
        message: String,
    ) -> Result<()> {
        self.require_phase(EvalPhase::AwaitingUserAction)?;
        if message.trim().is_empty() {
            return Err(EvalError::InvalidCommand(
                "user message is empty".to_string(),
            ));
        }
        if let Some(transition_id) = &transition {
            let transition = self.program.transition(transition_id)?;
            if transition.from != self.projection.current_node {
                return Err(EvalError::InvalidCommand(format!(
                    "transition {} does not start at current node",
                    transition_id.0
                )));
            }
            if self.projection.pending_transition.is_some() {
                return Err(EvalError::InvalidCommand(
                    "another transition is already pending".to_string(),
                ));
            }
            self.projection.pending_transition = Some(transition_id.clone());
        }
        self.record(EvalEventKind::UserMessage { message });
        self.projection.phase = EvalPhase::AwaitingAgentObservation;
        Ok(())
    }

    fn retry(&mut self, message: String) -> Result<()> {
        self.require_phase(EvalPhase::AwaitingSimulatorDecision)?;
        if self.projection.pending_transition.is_none() {
            return Err(EvalError::InvalidCommand(
                "cannot retry without a pending transition".to_string(),
            ));
        }
        self.send_retry_message(message)
    }

    fn send_retry_message(&mut self, message: String) -> Result<()> {
        if message.trim().is_empty() {
            return Err(EvalError::InvalidCommand(
                "retry message is empty".to_string(),
            ));
        }
        self.record(EvalEventKind::UserMessage { message });
        self.projection.phase = EvalPhase::AwaitingAgentObservation;
        Ok(())
    }

    fn commit(
        &mut self,
        transition_id: TransitionId,
        delivery: TransitionDelivery,
        evidence: Vec<EvidenceRef>,
    ) -> Result<()> {
        self.require_phase(EvalPhase::AwaitingSimulatorDecision)?;
        let transition = self.program.transition(&transition_id)?.clone();
        if transition.from != self.projection.current_node {
            return Err(EvalError::InvalidCommand(format!(
                "transition {} does not start at current node",
                transition_id.0
            )));
        }
        match delivery {
            TransitionDelivery::Explicit => {
                if self.projection.pending_transition.as_ref() != Some(&transition_id) {
                    return Err(EvalError::InvalidCommand(
                        "explicit commit does not match the pending transition".to_string(),
                    ));
                }
            }
            TransitionDelivery::Epsilon => {
                if self.projection.pending_transition.is_some()
                    || !self.program.epsilon_eligible(&transition)?
                {
                    return Err(EvalError::InvalidCommand(
                        "transition is not eligible for epsilon delivery".to_string(),
                    ));
                }
            }
        }

        let from = self.projection.current_node.clone();
        let previous_constraints = self.projection.constraints.clone();
        if let Err(error) = self.apply_node_constraints(&transition.to) {
            self.projection.constraints = previous_constraints;
            return Err(error);
        }
        self.projection.current_node = transition.to.clone();
        self.projection.pending_transition = None;
        self.projection.phase = EvalPhase::AwaitingUserAction;
        if self.program.node(&transition.to)?.terminal {
            self.projection.status = EvalVmStatus::Completed;
        }
        self.record(EvalEventKind::TransitionCommitted {
            transition: Some(transition_id),
            from,
            to: transition.to,
            delivery,
            evidence,
        });
        Ok(())
    }

    fn apply_node_constraints(&mut self, node_id: &NodeId) -> Result<()> {
        let node = self.program.node(node_id)?.clone();
        for delta in node.constraints {
            self.apply_constraint(node_id, delta)?;
        }
        Ok(())
    }

    fn apply_constraint(&mut self, node: &NodeId, delta: ConstraintDelta) -> Result<()> {
        match &delta.operation {
            ConstraintOperation::Add { id, value, .. } => {
                if self
                    .projection
                    .constraints
                    .active
                    .insert(id.clone(), value.clone())
                    .is_some()
                {
                    return Err(EvalError::InvalidCommand(format!(
                        "constraint already active: {}",
                        id.0
                    )));
                }
            }
            ConstraintOperation::Replace { target, id, value } => {
                if self.projection.constraints.active.remove(target).is_none() {
                    return Err(EvalError::InvalidCommand(format!(
                        "cannot replace inactive constraint: {}",
                        target.0
                    )));
                }
                self.projection
                    .constraints
                    .active
                    .insert(id.clone(), value.clone());
            }
            ConstraintOperation::Remove { target } => {
                if self.projection.constraints.active.remove(target).is_none() {
                    return Err(EvalError::InvalidCommand(format!(
                        "cannot remove inactive constraint: {}",
                        target.0
                    )));
                }
            }
        }
        self.projection
            .constraints
            .applied
            .push(ConstraintApplication {
                node: node.clone(),
                delta,
            });
        Ok(())
    }

    fn require_phase(&self, expected: EvalPhase) -> Result<()> {
        if self.projection.status != EvalVmStatus::Running {
            return Err(EvalError::InvalidCommand(
                "evaluation is not running".to_string(),
            ));
        }
        if self.projection.phase != expected {
            return Err(EvalError::InvalidCommand(format!(
                "command requires phase {expected:?}, current phase is {:?}",
                self.projection.phase
            )));
        }
        Ok(())
    }

    fn record(&mut self, kind: EvalEventKind) {
        self.projection.step = self.projection.step.saturating_add(1);
        self.events.push(EvalEvent {
            sequence: self.projection.step,
            kind,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{
        ActivationPolicy, ConstraintOperation, TaskCase, TaskNode, TaskTransition,
    };
    use crate::roles::{AgentObservationStatus, SimulatedUserCommand};
    use serde_json::json;

    fn program() -> EvalProgram {
        TaskCase {
            id: "case".to_string(),
            version: "1".to_string(),
            start: NodeId::from("start"),
            nodes: vec![
                TaskNode {
                    id: NodeId::from("start"),
                    constraints: vec![ConstraintDelta {
                        operation: ConstraintOperation::Add {
                            id: ConstraintId::from("kind"),
                            value: json!("phone"),
                            activation: ActivationPolicy::ExplicitDisclosure,
                        },
                        provenance: None,
                    }],
                    result_obligation: None,
                    evidence_obligation: None,
                    terminal: false,
                },
                TaskNode {
                    id: NodeId::from("done"),
                    constraints: vec![],
                    result_obligation: None,
                    evidence_obligation: None,
                    terminal: true,
                },
            ],
            transitions: vec![TaskTransition {
                id: TransitionId::from("finish"),
                from: NodeId::from("start"),
                to: NodeId::from("done"),
                kind: crate::program::TransitionKind::Progress,
                user_message: Some("finish".to_string()),
            }],
        }
        .compile()
        .expect("program")
    }

    #[test]
    fn vm_requires_observation_before_commit() {
        let mut vm = EvalVm::reducer(program()).expect("vm");
        vm.apply_command(SimulatedUserCommand::SendUserMessage {
            transition: Some(TransitionId::from("finish")),
            message: "finish".to_string(),
        })
        .expect("message");
        let error = vm
            .apply_command(SimulatedUserCommand::Commit {
                transition: TransitionId::from("finish"),
                delivery: TransitionDelivery::Explicit,
                evidence: vec![],
                reason: "done".to_string(),
            })
            .expect_err("observation required");
        assert!(error.to_string().contains("AwaitingSimulatorDecision"));
    }

    #[test]
    fn vm_commits_explicit_transition_after_observation() {
        let mut vm = EvalVm::reducer(program()).expect("vm");
        vm.apply_command(SimulatedUserCommand::SendUserMessage {
            transition: Some(TransitionId::from("finish")),
            message: "finish".to_string(),
        })
        .expect("message");
        vm.observe_agent(AgentObservation {
            status: AgentObservationStatus::Completed,
            assistant_text: "done".to_string(),
            events: vec![],
        })
        .expect("observation");
        vm.apply_command(SimulatedUserCommand::Commit {
            transition: TransitionId::from("finish"),
            delivery: TransitionDelivery::Explicit,
            evidence: vec![],
            reason: "done".to_string(),
        })
        .expect("commit");
        assert_eq!(vm.projection().status, EvalVmStatus::Completed);
        assert_eq!(vm.projection().current_node, NodeId::from("done"));
    }
}
