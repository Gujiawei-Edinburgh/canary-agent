//! Business-agnostic evaluation environments for tool-using agents.
//!
//! An [`EvalEnvironment`] owns task state and produces observations. An
//! [`EvaluatedPolicy`] turns each observation into an [`AgentAction`]. The
//! [`EvalRunner`] drives that interaction and sends the factual trajectory to
//! a [`Referee`] after the environment terminates.

mod environment;
mod error;
mod eval_command_tool;
mod program;
mod roles;
mod runner;

pub use environment::{
    ConstraintApplication, ConstraintLedger, EnvironmentController, EnvironmentControllerInput,
    EnvironmentDecision, EnvironmentEvent, EnvironmentEventKind, EnvironmentFuture,
    EnvironmentObservation, EnvironmentOutput, EnvironmentSnapshot, EnvironmentState,
    EnvironmentStatus, EvalEnvironment, EvidenceRef, GraphEnvironment, ObservationCause,
    ObservationContent, ObservationRealizer, ObservationRealizerInput, TerminationReason,
    VisibilityChange, VisibilityState,
};
pub use error::{EvalError, Result};
pub use eval_command_tool::{EnvironmentDecisionSink, EnvironmentDecisionTool};
pub use program::{
    ActivationPolicy, ConstraintDelta, ConstraintId, ConstraintOperation, NodeId, Obligation,
    TaskCase, TaskGraph, TaskNode, TaskTransition, TransitionId, TransitionKind,
};
pub use roles::{
    ActionFuture, AgentAction, AgentActionEvent, AgentActionStatus, EvalMetric, EvalReport,
    EvalReportFuture, EvaluatedPolicy, MetricResult, Referee, RefereeInput, RuntimeAgentPolicy,
};
pub use runner::{EvalRunner, EvalRunnerComponents, EvalRunnerConfig};
