use crate::error::Result;
use crate::ids::{BranchRef, RevisionId};
use crate::revision::AgentRevision;

/// Persistence contract for immutable revisions and mutable branch refs.
pub trait RevisionStore: Send + Sync {
    fn load_revision(&self, id: &RevisionId) -> Result<Option<AgentRevision>>;

    fn save_revision(&self, revision: &AgentRevision) -> Result<()>;

    fn branch_head(&self, branch: &BranchRef) -> Result<Option<RevisionId>>;

    fn compare_and_set_branch(
        &self,
        branch: &BranchRef,
        expected: Option<&RevisionId>,
        next: &RevisionId,
    ) -> Result<bool>;
}
