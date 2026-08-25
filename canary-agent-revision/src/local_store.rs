use crate::error::{canonical_json, RevisionError};
use crate::ids::{BranchRef, RevisionId};
use crate::revision::AgentRevision;
use crate::store::{RevisionFuture, RevisionStore};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

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
    pub async fn open(root: impl Into<PathBuf>) -> crate::Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("objects"))
            .await
            .map_err(|error| RevisionError::Store(error.to_string()))?;
        fs::create_dir_all(root.join("refs").join("heads"))
            .await
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

    async fn read_branch_unlocked(&self, branch: &BranchRef) -> crate::Result<Option<RevisionId>> {
        let path = self.branch_path(branch);
        let raw = match fs::read_to_string(path).await {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(RevisionError::Store(error.to_string())),
        };
        let value = raw.trim();
        if value.is_empty() {
            return Err(RevisionError::Store(format!(
                "branch {} has an empty head",
                branch
            )));
        }
        Ok(Some(RevisionId(value.to_string())))
    }

    async fn atomic_write(&self, path: &Path, contents: &[u8]) -> crate::Result<()> {
        let parent = path
            .parent()
            .ok_or_else(|| RevisionError::Store("path has no parent".to_string()))?;
        fs::create_dir_all(parent)
            .await
            .map_err(|error| RevisionError::Store(error.to_string()))?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| RevisionError::Store(error.to_string()))?
            .as_nanos();
        let temporary = parent.join(format!(".tmp-{}-{timestamp}", std::process::id()));
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .await
            .map_err(|error| RevisionError::Store(error.to_string()))?;
        file.write_all(contents)
            .await
            .map_err(|error| RevisionError::Store(error.to_string()))?;
        file.sync_all()
            .await
            .map_err(|error| RevisionError::Store(error.to_string()))?;
        fs::rename(temporary, path)
            .await
            .map_err(|error| RevisionError::Store(error.to_string()))
    }
}

impl RevisionStore for LocalRevisionStore {
    fn load_revision<'a>(
        &'a self,
        id: &'a RevisionId,
    ) -> RevisionFuture<'a, Option<AgentRevision>> {
        Box::pin(async move {
            let _guard = self.lock.lock().await;
            let contents = match fs::read(self.object_path(id)).await {
                Ok(contents) => contents,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(error) => return Err(RevisionError::Store(error.to_string())),
            };
            serde_json::from_slice(&contents)
                .map(Some)
                .map_err(|error| RevisionError::Serialization(error.to_string()))
        })
    }

    fn save_revision<'a>(&'a self, revision: &'a AgentRevision) -> RevisionFuture<'a, ()> {
        Box::pin(async move {
            let _guard = self.lock.lock().await;
            let bytes = canonical_json(revision)?;
            let path = self.object_path(&revision.revision_id);
            match fs::read(&path).await {
                Ok(existing) => {
                    if existing != bytes {
                        return Err(RevisionError::Store(format!(
                            "revision object {} already contains different data",
                            revision.revision_id.0
                        )));
                    }
                    Ok(())
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    self.atomic_write(&path, &bytes).await
                }
                Err(error) => Err(RevisionError::Store(error.to_string())),
            }
        })
    }

    fn branch_head<'a>(&'a self, branch: &'a BranchRef) -> RevisionFuture<'a, Option<RevisionId>> {
        Box::pin(async move {
            let _guard = self.lock.lock().await;
            self.read_branch_unlocked(branch).await
        })
    }

    fn compare_and_set_branch<'a>(
        &'a self,
        branch: &'a BranchRef,
        expected: Option<&'a RevisionId>,
        next: &'a RevisionId,
    ) -> RevisionFuture<'a, bool> {
        Box::pin(async move {
            let _guard = self.lock.lock().await;
            let actual = self.read_branch_unlocked(branch).await?;
            if actual.as_ref() != expected {
                return Ok(false);
            }
            self.atomic_write(&self.branch_path(branch), next.0.as_bytes())
                .await?;
            Ok(true)
        })
    }
}
