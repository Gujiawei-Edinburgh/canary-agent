use crate::diff::AgentDiff;
use crate::error::{canonical_json, Result, RevisionError};
use crate::ids::{AgentId, RevisionId, SpecDigest};
use crate::spec::AgentSpec;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitMessage {
    pub subject: String,
    pub body: Option<String>,
}

impl CommitMessage {
    pub fn new(subject: impl Into<String>) -> Result<Self> {
        Self::with_body(subject, None::<String>)
    }

    pub fn with_body(subject: impl Into<String>, body: Option<impl Into<String>>) -> Result<Self> {
        let subject = subject.into();
        if subject.trim().is_empty() {
            return Err(RevisionError::InvalidCommitMessage(
                "subject is empty".to_string(),
            ));
        }
        if subject.contains('\n') || subject.contains('\r') {
            return Err(RevisionError::InvalidCommitMessage(
                "subject must be a single line".to_string(),
            ));
        }
        Ok(Self {
            subject,
            body: body.map(Into::into),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevisionMetadata {
    pub author: Option<String>,
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentRevision {
    pub revision_id: RevisionId,
    pub agent_id: AgentId,
    pub spec_digest: SpecDigest,
    pub spec: AgentSpec,
    pub parents: Vec<RevisionId>,
    pub message: CommitMessage,
    pub metadata: RevisionMetadata,
}

impl AgentRevision {
    pub fn commit(
        spec: AgentSpec,
        parents: Vec<RevisionId>,
        message: CommitMessage,
        metadata: RevisionMetadata,
    ) -> Result<Self> {
        spec.validate()?;
        validate_parents(&parents)?;
        let spec_digest = spec.digest()?;
        let identity = RevisionIdentity {
            agent_id: &spec.agent_id,
            spec_digest: &spec_digest,
            parents: &parents,
            message: &message,
            metadata: &metadata,
        };
        let digest = Sha256::digest(canonical_json(&identity)?);
        let revision_id = RevisionId(format!(
            "sha256:{}",
            digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        ));
        Ok(Self {
            revision_id,
            agent_id: spec.agent_id.clone(),
            spec_digest,
            spec,
            parents,
            message,
            metadata,
        })
    }

    pub fn diff(&self, other: &Self) -> Result<AgentDiff> {
        self.spec.diff(&other.spec)
    }
}

#[derive(Serialize)]
struct RevisionIdentity<'a> {
    agent_id: &'a AgentId,
    spec_digest: &'a SpecDigest,
    parents: &'a [RevisionId],
    message: &'a CommitMessage,
    metadata: &'a RevisionMetadata,
}

fn validate_parents(parents: &[RevisionId]) -> Result<()> {
    let mut unique = BTreeSet::new();
    for parent in parents {
        if parent.0.trim().is_empty() {
            return Err(RevisionError::InvalidSpec(
                "parent revision ID is empty".to_string(),
            ));
        }
        if !unique.insert(parent) {
            return Err(RevisionError::InvalidSpec(
                "duplicate parent revision ID".to_string(),
            ));
        }
    }
    Ok(())
}
