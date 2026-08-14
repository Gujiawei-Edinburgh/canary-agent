use crate::environment::{EnvironmentDecision, EvidenceRef};
use crate::program::TransitionId;
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
                        "required": ["kind", "transition", "evidence", "reason"],
                        "additionalProperties": false
                    },
                    {
                        "properties": {
                            "kind": { "const": "retry" },
                            "reason": { "type": "string" }
                        },
                        "required": ["kind", "reason"],
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
                ]
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
        evidence: Vec<EvidenceRef>,
        reason: String,
    },
    Retry {
        reason: String,
    },
    Terminate {
        reason: String,
    },
}

impl EnvironmentDecisionRequest {
    fn into_decision(self) -> EnvironmentDecision {
        match self {
            Self::Transition {
                transition,
                evidence,
                reason,
            } => EnvironmentDecision::Transition {
                transition: TransitionId(transition),
                evidence,
                reason,
            },
            Self::Retry { reason } => EnvironmentDecision::Retry { reason },
            Self::Terminate { reason } => EnvironmentDecision::Terminate { reason },
        }
    }
}
