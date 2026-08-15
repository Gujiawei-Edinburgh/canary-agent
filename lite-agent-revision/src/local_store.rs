use crate::error::{canonical_json, Result, RevisionError};
use crate::ids::{BranchRef, RevisionId};
use crate::revision::AgentRevision;
use crate::store::RevisionStore;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// A small JSON filesystem store for local development and embedding.
///
/// Updates are serialized within this store instance and revision files are
/// written by rename. A multi-process deployment should provide a store with
/// process-wide compare-and-set semantics.
pub struct LocalRevisionStore {
    root: PathBuf,
    lock: Mutex<()>,
}

impl LocalRevisionStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("objects"))
            .map_err(|error| RevisionError::Store(error.to_string()))?;
        fs::create_dir_all(root.join("refs").join("heads"))
            .map_err(|error| RevisionError::Store(error.to_string()))?;
        Ok(Self {
            root,
            lock: Mutex::new(()),
        })
    }

    fn object_path(&self, id: &RevisionId) -> PathBuf {
        self.root
            .join("objects")
            .join(id.0.replace(':', "_"))
            .with_extension("json")
    }

    fn branch_path(&self, branch: &BranchRef) -> PathBuf {
        self.root
            .join("refs")
            .join("heads")
            .join(&branch.agent_id.0)
            .join(&branch.name.0)
    }

    fn read_branch_unlocked(&self, branch: &BranchRef) -> Result<Option<RevisionId>> {
        let path = self.branch_path(branch);
        if !path.exists() {
            return Ok(None);
        }
        let mut raw = String::new();
        File::open(path)
            .map_err(|error| RevisionError::Store(error.to_string()))?
            .read_to_string(&mut raw)
            .map_err(|error| RevisionError::Store(error.to_string()))?;
        let value = raw.trim();
        if value.is_empty() {
            return Err(RevisionError::Store(format!(
                "branch {} has an empty head",
                branch
            )));
        }
        Ok(Some(RevisionId(value.to_string())))
    }

    fn atomic_write(&self, path: &Path, contents: &[u8]) -> Result<()> {
        let parent = path
            .parent()
            .ok_or_else(|| RevisionError::Store("path has no parent".to_string()))?;
        fs::create_dir_all(parent).map_err(|error| RevisionError::Store(error.to_string()))?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| RevisionError::Store(error.to_string()))?
            .as_nanos();
        let temporary = parent.join(format!(".tmp-{}-{timestamp}", std::process::id()));
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| RevisionError::Store(error.to_string()))?;
        file.write_all(contents)
            .and_then(|_| file.sync_all())
            .map_err(|error| RevisionError::Store(error.to_string()))?;
        fs::rename(temporary, path).map_err(|error| RevisionError::Store(error.to_string()))
    }
}

impl RevisionStore for LocalRevisionStore {
    fn load_revision(&self, id: &RevisionId) -> Result<Option<AgentRevision>> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| RevisionError::Store("store lock poisoned".to_string()))?;
        let path = self.object_path(id);
        if !path.exists() {
            return Ok(None);
        }
        let contents = fs::read(path).map_err(|error| RevisionError::Store(error.to_string()))?;
        serde_json::from_slice(&contents)
            .map(Some)
            .map_err(|error| RevisionError::Serialization(error.to_string()))
    }

    fn save_revision(&self, revision: &AgentRevision) -> Result<()> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| RevisionError::Store("store lock poisoned".to_string()))?;
        let bytes = canonical_json(revision)?;
        let path = self.object_path(&revision.revision_id);
        if path.exists() {
            let existing =
                fs::read(&path).map_err(|error| RevisionError::Store(error.to_string()))?;
            if existing != bytes {
                return Err(RevisionError::Store(format!(
                    "revision object {} already contains different data",
                    revision.revision_id.0
                )));
            }
            return Ok(());
        }
        self.atomic_write(&path, &bytes)
    }

    fn branch_head(&self, branch: &BranchRef) -> Result<Option<RevisionId>> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| RevisionError::Store("store lock poisoned".to_string()))?;
        self.read_branch_unlocked(branch)
    }

    fn compare_and_set_branch(
        &self,
        branch: &BranchRef,
        expected: Option<&RevisionId>,
        next: &RevisionId,
    ) -> Result<bool> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| RevisionError::Store("store lock poisoned".to_string()))?;
        let actual = self.read_branch_unlocked(branch)?;
        if actual.as_ref() != expected {
            return Ok(false);
        }
        self.atomic_write(&self.branch_path(branch), next.0.as_bytes())?;
        Ok(true)
    }
}
