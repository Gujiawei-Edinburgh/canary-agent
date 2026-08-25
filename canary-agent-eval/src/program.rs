use crate::{EvalError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TransitionId(pub String);

impl From<&str> for TransitionId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConstraintId(pub String);

impl From<&str> for ConstraintId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub String);

impl From<&str> for NodeId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConstraintOperation {
    Add {
        id: ConstraintId,
        value: Value,
    },
    Replace {
        target: ConstraintId,
        id: ConstraintId,
        value: Value,
    },
    Remove {
        target: ConstraintId,
    },
}

impl ConstraintOperation {
    pub fn id(&self) -> Option<&ConstraintId> {
        match self {
            Self::Add { id, .. } | Self::Replace { id, .. } => Some(id),
            Self::Remove { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintDelta {
    pub operation: ConstraintOperation,
    pub provenance: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Obligation {
    pub id: String,
    pub payload: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionKind {
    Progress,
    Revision,
    BranchSelection,
    ConflictResolution,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskNode {
    pub id: NodeId,
    pub constraints: Vec<ConstraintDelta>,
    pub result_obligation: Option<Obligation>,
    pub evidence_obligation: Option<Obligation>,
    pub terminal: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskTransition {
    pub id: TransitionId,
    pub from: NodeId,
    pub to: NodeId,
    pub kind: TransitionKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCase {
    pub id: String,
    pub version: String,
    pub start: NodeId,
    pub nodes: Vec<TaskNode>,
    pub transitions: Vec<TaskTransition>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskGraph {
    pub case_id: String,
    pub version: String,
    pub start: NodeId,
    pub nodes: BTreeMap<NodeId, TaskNode>,
    pub transitions: BTreeMap<TransitionId, TaskTransition>,
}

impl TaskCase {
    /// Validate and lower an authored case into an immutable, indexed program.
    pub fn compile(self) -> Result<TaskGraph> {
        if self.id.trim().is_empty() {
            return Err(EvalError::InvalidTaskGraph("case id is empty".to_string()));
        }
        if self.nodes.is_empty() {
            return Err(EvalError::InvalidTaskGraph("case has no nodes".to_string()));
        }

        let mut nodes = BTreeMap::new();
        for node in self.nodes {
            if node.id.0.trim().is_empty() {
                return Err(EvalError::InvalidTaskGraph("node id is empty".to_string()));
            }
            if nodes.insert(node.id.clone(), node).is_some() {
                return Err(EvalError::InvalidTaskGraph("duplicate node id".to_string()));
            }
        }
        if !nodes.contains_key(&self.start) {
            return Err(EvalError::InvalidTaskGraph(format!(
                "start node does not exist: {}",
                self.start.0
            )));
        }

        let mut transitions = BTreeMap::new();
        for transition in self.transitions {
            if transition.id.0.trim().is_empty() {
                return Err(EvalError::InvalidTaskGraph(
                    "transition id is empty".to_string(),
                ));
            }
            if !nodes.contains_key(&transition.from) || !nodes.contains_key(&transition.to) {
                return Err(EvalError::InvalidTaskGraph(format!(
                    "transition {} references an unknown node",
                    transition.id.0
                )));
            }
            if transitions
                .insert(transition.id.clone(), transition)
                .is_some()
            {
                return Err(EvalError::InvalidTaskGraph(
                    "duplicate transition id".to_string(),
                ));
            }
        }

        validate_constraint_ids(&nodes)?;
        Ok(TaskGraph {
            case_id: self.id,
            version: self.version,
            start: self.start,
            nodes,
            transitions,
        })
    }
}

impl TaskGraph {
    pub fn outgoing(&self, node: &NodeId) -> Vec<&TaskTransition> {
        self.transitions
            .values()
            .filter(|transition| &transition.from == node)
            .collect()
    }

    pub fn transition(&self, id: &TransitionId) -> Result<&TaskTransition> {
        self.transitions.get(id).ok_or_else(|| {
            EvalError::InvalidEnvironmentAction(format!("unknown transition: {}", id.0))
        })
    }

    pub fn node(&self, id: &NodeId) -> Result<&TaskNode> {
        self.nodes
            .get(id)
            .ok_or_else(|| EvalError::InvalidTaskGraph(format!("unknown node: {}", id.0)))
    }
}

fn validate_constraint_ids(nodes: &BTreeMap<NodeId, TaskNode>) -> Result<()> {
    let mut declared = BTreeSet::new();
    let mut targets = Vec::new();
    for node in nodes.values() {
        for delta in &node.constraints {
            if let Some(id) = delta.operation.id()
                && !declared.insert(id.clone())
            {
                return Err(EvalError::InvalidTaskGraph(format!(
                    "duplicate constraint id: {}",
                    id.0
                )));
            }
            if let ConstraintOperation::Replace { target, .. } = &delta.operation {
                targets.push(target.clone());
            }
        }
    }
    for target in targets {
        if !declared.contains(&target) {
            return Err(EvalError::InvalidTaskGraph(format!(
                "replacement targets unknown constraint: {}",
                target.0
            )));
        }
    }
    Ok(())
}
