use crate::events::TokenUsage;
use crate::projection::ChatMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelRequest {
    pub messages: Vec<ChatMessage>,
    pub functions: Vec<FunctionSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelResponse {
    Assistant {
        text: Option<String>,
        function_calls: Vec<ModelFunctionCall>,
    },
    AssistantMessage {
        text: String,
    },
    FunctionCalls {
        calls: Vec<ModelFunctionCall>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelFunctionCall {
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelStreamEvent {
    /// One provider chunk containing nonempty text or tool-call output.
    /// Emit before presentation events, once per chunk (not once per token).
    /// Excludes role-only, usage-only, and termination chunks.
    OutputProgress,
    AssistantDelta {
        text: String,
    },
    TokenUsage {
        usage: TokenUsage,
    },
}
