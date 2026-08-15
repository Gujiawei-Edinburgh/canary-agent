//! Static, Git-like revision control for agent compositions.
//!
//! This crate stores and compares declared agent specifications. It does not
//! execute agents or evaluate runtime behavior.

mod controller;
mod diff;
mod error;
mod ids;
mod local_store;
mod merge;
mod revision;
mod spec;
mod store;

pub use controller::{AgentCheckout, RevisionController};
pub use diff::{AgentDiff, ToolChange, ToolDiff, ValueChange};
pub use error::{Result, RevisionError};
pub use ids::{AgentId, BranchName, BranchRef, RevisionId, SpecDigest};
pub use local_store::LocalRevisionStore;
pub use merge::{MergeConflict, MergeResult};
pub use revision::{AgentRevision, CommitMessage, RevisionMetadata};
pub use spec::{
    AgentSpec, ComponentRef, ModelSpec, PromptSpec, RuntimePolicySpec, ToolInterfaceSpec, ToolSpec,
};
pub use store::RevisionStore;
