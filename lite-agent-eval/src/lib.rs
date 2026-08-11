//! Business-agnostic evaluation primitives for tool-using agents.
//!
//! The crate separates four concerns:
//!
//! - a source [`TaskCase`] compiled into an immutable [`EvalProgram`];
//! - an [`EvalVm`] that owns execution state and the factual trajectory;
//! - a simulated-user processor that controls task progression through typed
//!   [`SimulatedUserCommand`] values;
//! - a post-run [`Referee`] that evaluates the program and trajectory.
//!
//! Constraint payloads, guard semantics, simulated-user policy, and referee
//! metrics are intentionally opaque to this crate. Hosts provide those parts.

mod error;
mod program;
mod roles;
mod eval_command_tool;
mod vm;

pub use error::{EvalError, Result};
pub use program::{
    ActivationPolicy, ConstraintDelta, ConstraintId, ConstraintOperation, EvalProgram, NodeId,
    Obligation, TaskCase, TaskNode, TaskTransition, TransitionId, TransitionKind,
};
pub use roles::{
    AgentInput, AgentObservation, AgentObservationEvent, AgentObservationStatus, AgentRoleIo,
    AgentRoleOutput, EvalMetric, EvalReport, MetricReferee, MetricResult, ProcessorInput, Referee,
    RefereeInput, RoleFuture, RuntimeAgentIo, SimulatedUserCommand, SimulatedUserProcessor,
    TestedAgentIo,
};
pub use eval_command_tool::{EvalCommandSink, EvalCommandTool};
pub use vm::{
    ConstraintApplication, ConstraintLedger, EvalEvent, EvalEventKind, EvalPhase, EvalProjection,
    EvalVm, EvalVmComponents, EvalVmStatus, EvidenceRef, TransitionDelivery,
};
