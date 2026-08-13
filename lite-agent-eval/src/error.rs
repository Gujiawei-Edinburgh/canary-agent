use thiserror::Error;

pub type Result<T> = std::result::Result<T, EvalError>;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("invalid task graph: {0}")]
    InvalidTaskGraph(String),

    #[error("invalid environment action: {0}")]
    InvalidEnvironmentAction(String),

    #[error("environment failed: {0}")]
    Environment(String),

    #[error("evaluation role failed: {0}")]
    Role(String),

    #[error("agent runtime failed: {0}")]
    Agent(#[from] lite_agent_runtime::AgentError),

    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}
