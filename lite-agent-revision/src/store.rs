use crate::error::Result;
use crate::ids::{BranchRef, RevisionId};
use crate::revision::AgentRevision;
use std::future::Future;
use std::pin::Pin;

pub type RevisionFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

/// Persistence contract for immutable revisions and mutable branch refs.
pub trait RevisionStore: Send + Sync {
    fn load_revision<'a>(&'a self, id: &'a RevisionId)
        -> RevisionFuture<'a, Option<AgentRevision>>;

    fn save_revision<'a>(&'a self, revision: &'a AgentRevision) -> RevisionFuture<'a, ()>;

    fn branch_head<'a>(&'a self, branch: &'a BranchRef) -> RevisionFuture<'a, Option<RevisionId>>;

    fn compare_and_set_branch<'a>(
        &'a self,
        branch: &'a BranchRef,
        expected: Option<&'a RevisionId>,
        next: &'a RevisionId,
    ) -> RevisionFuture<'a, bool>;
}
