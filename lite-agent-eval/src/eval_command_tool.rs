use crate::environment::{EnvironmentDecision, EvidenceRef, VisibilityChange};
use crate::program::{ConstraintId, TransitionId};
use lite_agent_runtime::{
    AgentFunction, DiscardResolver, FunctionContext, FunctionExecution, FunctionLimits,
    FunctionOutputResolver, FunctionRecoveryPolicy, FunctionSpec, Result as AgentResult,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Default)]
pub struct EnvironmentDecisionSink {
    decision: Mutex<Option<EnvironmentDecision>>,
}

impl EnvironmentDecisionSink {
    pub fn take(&self) -> Option<EnvironmentDecision> {
        self.decision.lock().ok()?.take()
    }

    fn submit(&self, decision: EnvironmentDecision) -> Result<(), String> {
        let mut pending = self
            .decision
            .lock()
            .map_err(|_| "environment decision sink is poisoned".to_string())?;
        if pending.as_ref() == Some(&decision) {
            return Ok(());
        }
        if pending.is_some() {
            return Err("a different environment decision is already pending".to_string());
        }
        *pending = Some(decision);
        Ok(())
    }
}

pub struct EnvironmentDecisionTool {
    sink: Arc<EnvironmentDecisionSink>,
}

impl EnvironmentDecisionTool {
    pub fn new(sink: Arc<EnvironmentDecisionSink>) -> Self {
        Self { sink }
    }
}

impl AgentFunction for EnvironmentDecisionTool {
    fn spec(&self) -> FunctionSpec {
        FunctionSpec {
            name: "environment_decision".to_string(),
            description: "Submit exactly one typed environment decision after inspecting the evaluated policy's action.".to_string(),
            parameters: json!({
                "type": "object",
                "oneOf": [
                    {
                        "properties": {
                            "kind": { "const": "transition" },
                            "transition": { "type": "string" },
                            "visibility": { "$ref": "#/$defs/visibility_changes" },
                            "evidence": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "kind": { "type": "string" },
                                        "reference": { "type": "string" }
                                    },
                                    "required": ["kind", "reference"],
                                    "additionalProperties": false
                                }
                            },
                            "reason": { "type": "string" }
                        },
                        "required": ["kind", "transition", "visibility", "evidence", "reason"],
                        "additionalProperties": false
                    },
                    {
                        "properties": {
                            "kind": { "const": "retry" },
                            "visibility": { "$ref": "#/$defs/visibility_changes" },
                            "reason": { "type": "string" }
                        },
                        "required": ["kind", "visibility", "reason"],
                        "additionalProperties": false
                    },
                    {
                        "properties": {
                            "kind": { "const": "terminate" },
                            "reason": { "type": "string" }
                        },
                        "required": ["kind", "reason"],
                        "additionalProperties": false
                    }
                ],
                "$defs": {
                    "visibility_changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": { "enum": ["disclose", "derive", "conceal"] },
                                "constraint": { "type": "string" }
                            },
                            "required": ["kind", "constraint"],
                            "additionalProperties": false
                        }
                    }
                }
            }),
        }
    }

    fn limits(&self) -> FunctionLimits {
        FunctionLimits {
            time_budget: Duration::from_secs(5),
            max_output_bytes: 16 * 1024,
        }
    }

    fn recovery_policy(&self) -> FunctionRecoveryPolicy {
        FunctionRecoveryPolicy::Idempotent
    }

    fn output_resolver(&self) -> &dyn FunctionOutputResolver {
        &DiscardResolver
    }

    fn call<'a>(
        &'a self,
        args: Value,
        _context: FunctionContext,
    ) -> Pin<Box<dyn Future<Output = AgentResult<FunctionExecution>> + Send + 'a>> {
        Box::pin(async move {
            let request: EnvironmentDecisionRequest =
                serde_json::from_value(args).map_err(|error| {
                    lite_agent_runtime::AgentError::InvalidFunctionArguments {
                        name: "environment_decision".to_string(),
                        message: error.to_string(),
                    }
                })?;
            self.sink
                .submit(request.into_decision())
                .map_err(
                    |message| lite_agent_runtime::AgentError::InvalidFunctionArguments {
                        name: "environment_decision".to_string(),
                        message,
                    },
                )?;
            Ok(FunctionExecution::Completed {
                output: json!({"accepted": true}),
            })
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum EnvironmentDecisionRequest {
    Transition {
        transition: String,
        visibility: Vec<VisibilityChangeRequest>,
        evidence: Vec<EvidenceRef>,
        reason: String,
    },
    Retry {
        visibility: Vec<VisibilityChangeRequest>,
        reason: String,
    },
    Terminate {
        reason: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum VisibilityChangeRequest {
    Disclose { constraint: String },
    Derive { constraint: String },
    Conceal { constraint: String },
}

impl EnvironmentDecisionRequest {
    fn into_decision(self) -> EnvironmentDecision {
        match self {
            Self::Transition {
                transition,
                visibility,
                evidence,
                reason,
            } => EnvironmentDecision::Transition {
                transition: TransitionId(transition),
                visibility: visibility
                    .into_iter()
                    .map(VisibilityChangeRequest::into_change)
                    .collect(),
                evidence,
                reason,
            },
            Self::Retry { visibility, reason } => EnvironmentDecision::Retry {
                visibility: visibility
                    .into_iter()
                    .map(VisibilityChangeRequest::into_change)
                    .collect(),
                reason,
            },
            Self::Terminate { reason } => EnvironmentDecision::Terminate { reason },
        }
    }
}

impl VisibilityChangeRequest {
    fn into_change(self) -> VisibilityChange {
        match self {
            Self::Disclose { constraint } => VisibilityChange::Disclose {
                constraint: ConstraintId(constraint),
            },
            Self::Derive { constraint } => VisibilityChange::Derive {
                constraint: ConstraintId(constraint),
            },
            Self::Conceal { constraint } => VisibilityChange::Conceal {
                constraint: ConstraintId(constraint),
            },
        }
    }
}
