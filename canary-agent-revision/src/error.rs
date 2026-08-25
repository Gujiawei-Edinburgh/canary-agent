use std::fmt;

pub type Result<T> = std::result::Result<T, RevisionError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RevisionError {
    InvalidSpec(String),
    InvalidCommitMessage(String),
    InvalidBranch(String),
    Serialization(String),
    Store(String),
    RevisionNotFound(String),
    BranchNotFound(String),
    BranchAlreadyExists(String),
    ConcurrentUpdate(String),
    NoCommonAncestor(String),
    NothingToMerge(String),
    MergeConflicts(Vec<String>),
}

impl fmt::Display for RevisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpec(message) => write!(formatter, "invalid agent spec: {message}"),
            Self::InvalidCommitMessage(message) => {
                write!(formatter, "invalid commit message: {message}")
            }
            Self::InvalidBranch(message) => write!(formatter, "invalid branch: {message}"),
            Self::Serialization(message) => write!(formatter, "serialization failed: {message}"),
            Self::Store(message) => write!(formatter, "revision store failed: {message}"),
            Self::RevisionNotFound(id) => write!(formatter, "revision not found: {id}"),
            Self::BranchNotFound(branch) => write!(formatter, "branch not found: {branch}"),
            Self::BranchAlreadyExists(branch) => {
                write!(formatter, "branch already exists: {branch}")
            }
            Self::ConcurrentUpdate(branch) => {
                write!(formatter, "branch changed concurrently: {branch}")
            }
            Self::NoCommonAncestor(branches) => {
                write!(formatter, "branches have no common ancestor: {branches}")
            }
            Self::NothingToMerge(branches) => write!(formatter, "nothing to merge: {branches}"),
            Self::MergeConflicts(paths) => {
                write!(formatter, "merge has conflicts at: {}", paths.join(", "))
            }
        }
    }
}

impl std::error::Error for RevisionError {}

pub(crate) fn canonical_json<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|error| RevisionError::Serialization(error.to_string()))
}
