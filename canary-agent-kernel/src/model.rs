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
pub struct ModelContinuation {
    /// Versioned adapter-specific format; interpreted only by a matching adapter.
    pub format: String,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelResponse {
    Assistant {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        continuation: Option<ModelContinuation>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TurnItemKind;
    use serde_json::json;

    #[test]
    fn legacy_model_records_default_to_no_continuation() {
        let record = json!({"type": "model_response", "text": "hello", "function_calls": []});
        let decoded: TurnItemKind = serde_json::from_value(record.clone()).unwrap();
        assert!(matches!(
            &decoded,
            TurnItemKind::ModelResponse {
                continuation: None,
                ..
            }
        ));
        assert_eq!(serde_json::to_value(decoded).unwrap(), record);
        let response: ModelResponse = serde_json::from_value(json!({
            "Assistant": {"text": "hello", "function_calls": []}
        }))
        .unwrap();
        assert!(matches!(
            response,
            ModelResponse::Assistant {
                continuation: None,
                ..
            }
        ));
    }
}
