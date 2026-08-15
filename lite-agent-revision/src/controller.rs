use crate::error::{Result, RevisionError};
use crate::ids::{BranchRef, RevisionId};
use crate::merge::{choose_merge_base, collect_ancestors, merge_specs, MergeResult};
use crate::revision::{AgentRevision, CommitMessage, RevisionMetadata};
use crate::spec::AgentSpec;
use crate::store::RevisionStore;
use std::sync::Arc;

pub struct RevisionController<S> {
    store: Arc<S>,
}

impl<S> RevisionController<S>
where
    S: RevisionStore + 'static,
{
    pub fn new(store: S) -> Self {
        Self {
            store: Arc::new(store),
        }
    }

    pub fn store(&self) -> Arc<S> {
        self.store.clone()
    }

    pub fn commit(
        &self,
        branch: &BranchRef,
        spec: AgentSpec,
        message: CommitMessage,
    ) -> Result<AgentRevision> {
        if branch.agent_id != spec.agent_id {
            return Err(RevisionError::InvalidSpec(format!(
                "branch {} belongs to {}, but spec belongs to {}",
                branch.name.0, branch.agent_id.0, spec.agent_id.0
            )));
        }
        let head = self.store.branch_head(branch)?;
        let parents = head.iter().cloned().collect::<Vec<_>>();
        let revision = AgentRevision::commit(
            spec,
            parents,
            message,
            RevisionMetadata {
                author: None,
                created_at: None,
            },
        )?;
        self.store.save_revision(&revision)?;
        if !self
            .store
            .compare_and_set_branch(branch, head.as_ref(), &revision.revision_id)?
        {
            return Err(RevisionError::ConcurrentUpdate(branch.to_string()));
        }
        Ok(revision)
    }

    pub fn create_branch(&self, source: &BranchRef, target: BranchRef) -> Result<RevisionId> {
        if source.agent_id != target.agent_id {
            return Err(RevisionError::InvalidBranch(
                "source and target branches belong to different agents".to_string(),
            ));
        }
        let source_head = self
            .store
            .branch_head(source)?
            .ok_or_else(|| RevisionError::BranchNotFound(source.to_string()))?;
        if self.store.branch_head(&target)?.is_some() {
            return Err(RevisionError::BranchAlreadyExists(target.to_string()));
        }
        if !self
            .store
            .compare_and_set_branch(&target, None, &source_head)?
        {
            return Err(RevisionError::BranchAlreadyExists(target.to_string()));
        }
        Ok(source_head)
    }

    pub fn checkout(&self, branch: &BranchRef) -> Result<AgentCheckout> {
        let revision_id = self
            .store
            .branch_head(branch)?
            .ok_or_else(|| RevisionError::BranchNotFound(branch.to_string()))?;
        let revision = self
            .store
            .load_revision(&revision_id)?
            .ok_or_else(|| RevisionError::RevisionNotFound(revision_id.0.clone()))?;
        Ok(AgentCheckout {
            branch: branch.clone(),
            revision,
        })
    }

    pub fn load_revision(&self, id: &RevisionId) -> Result<AgentRevision> {
        self.store
            .load_revision(id)?
            .ok_or_else(|| RevisionError::RevisionNotFound(id.0.clone()))
    }

    pub fn prepare_merge(&self, source: &BranchRef, target: &BranchRef) -> Result<MergeResult> {
        self.validate_branch_pair(source, target)?;
        let source_head = self.branch_head(source)?;
        let target_head = self.branch_head(target)?;
        if source_head == target_head {
            return Err(RevisionError::NothingToMerge(source.to_string()));
        }
        let base = self.find_merge_base(&target_head, &source_head)?;
        let ours = self.load_revision(&target_head)?;
        let theirs = self.load_revision(&source_head)?;
        let base_revision = self.load_revision(&base)?;
        merge_specs(
            base,
            target_head,
            source_head,
            &base_revision.spec,
            &ours.spec,
            &theirs.spec,
        )
    }

    pub fn merge(
        &self,
        source: &BranchRef,
        target: &BranchRef,
        message: CommitMessage,
    ) -> Result<AgentRevision> {
        let result = self.prepare_merge(source, target)?;
        if !result.is_clean() {
            return Err(RevisionError::MergeConflicts(
                result
                    .conflicts
                    .iter()
                    .map(|conflict| conflict.path.clone())
                    .collect(),
            ));
        }
        self.commit_merge(
            target,
            &result.theirs,
            result.spec.expect("clean merge spec"),
            message,
        )
    }

    pub fn commit_merge(
        &self,
        target: &BranchRef,
        source_revision: &RevisionId,
        spec: AgentSpec,
        message: CommitMessage,
    ) -> Result<AgentRevision> {
        let target_head = self.branch_head(target)?;
        let target_parent = target_head.clone();
        if spec.agent_id != target.agent_id {
            return Err(RevisionError::InvalidSpec(format!(
                "merge spec belongs to {}, but target branch belongs to {}",
                spec.agent_id.0, target.agent_id.0
            )));
        }
        let source = self.load_revision(source_revision)?;
        if source.agent_id != target.agent_id {
            return Err(RevisionError::InvalidSpec(
                "merge source belongs to a different agent".to_string(),
            ));
        }
        let revision = AgentRevision::commit(
            spec,
            vec![target_parent, source_revision.clone()],
            message,
            RevisionMetadata {
                author: None,
                created_at: None,
            },
        )?;
        self.store.save_revision(&revision)?;
        if !self
            .store
            .compare_and_set_branch(target, Some(&target_head), &revision.revision_id)?
        {
            return Err(RevisionError::ConcurrentUpdate(target.to_string()));
        }
        Ok(revision)
    }

    pub fn parents(&self, id: &RevisionId) -> Result<Vec<AgentRevision>> {
        let revision = self.load_revision(id)?;
        revision
            .parents
            .iter()
            .map(|parent| self.load_revision(parent))
            .collect()
    }

    fn branch_head(&self, branch: &BranchRef) -> Result<RevisionId> {
        self.store
            .branch_head(branch)?
            .ok_or_else(|| RevisionError::BranchNotFound(branch.to_string()))
    }

    fn validate_branch_pair(&self, source: &BranchRef, target: &BranchRef) -> Result<()> {
        if source.agent_id != target.agent_id {
            return Err(RevisionError::InvalidBranch(
                "source and target branches belong to different agents".to_string(),
            ));
        }
        Ok(())
    }

    fn find_merge_base(&self, ours: &RevisionId, theirs: &RevisionId) -> Result<RevisionId> {
        let mut ancestor_maps = std::collections::BTreeMap::new();
        ancestor_maps.insert(
            ours.clone(),
            collect_ancestors(ours, |id| Ok(self.load_revision(id)?.parents))?,
        );
        ancestor_maps.insert(
            theirs.clone(),
            collect_ancestors(theirs, |id| Ok(self.load_revision(id)?.parents))?,
        );
        choose_merge_base(ours, theirs, &ancestor_maps)
            .ok_or_else(|| RevisionError::NoCommonAncestor(format!("{} and {}", ours.0, theirs.0)))
    }
}

#[derive(Debug, Clone)]
pub struct AgentCheckout {
    pub branch: BranchRef,
    pub revision: AgentRevision,
}
