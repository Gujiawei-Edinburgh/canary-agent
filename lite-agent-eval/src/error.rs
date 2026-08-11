use thiserror::Error;

pub type Result<T> = std::result::Result<T, EvalError>;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("invalid evaluation program: {0}")]
    InvalidProgram(String),

    #[error("invalid VM command: {0}")]
    InvalidCommand(String),

    #[error("evaluation role failed: {0}")]
    Role(String),

    #[error("agent runtime failed: {0}")]
    Agent(#[from] lite_agent_runtime::AgentError),

    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}
