use crate::error::{Result, RevisionError};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_component("agent id", &value)?;
        Ok(Self(value))
    }
}

impl From<&str> for AgentId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BranchName(pub String);

impl BranchName {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_component("branch name", &value)?;
        Ok(Self(value))
    }
}

impl From<&str> for BranchName {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BranchRef {
    pub agent_id: AgentId,
    pub name: BranchName,
}

impl BranchRef {
    pub fn new(agent_id: AgentId, name: BranchName) -> Self {
        Self { agent_id, name }
    }
}

impl fmt::Display for BranchRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.agent_id.0, self.name.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RevisionId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SpecDigest(pub String);

pub(crate) fn validate_component(kind: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(RevisionError::InvalidBranch(format!("{kind} is empty")));
    }
    if value == "." || value == ".." || value.contains('/') || value.contains('\\') {
        return Err(RevisionError::InvalidBranch(format!(
            "{kind} contains a path separator or reserved component"
        )));
    }
    Ok(())
}
