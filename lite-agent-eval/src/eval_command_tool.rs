use crate::program::TransitionId;
use crate::roles::SimulatedUserCommand;
use crate::vm::{EvidenceRef, TransitionDelivery};
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
pub struct EvalCommandSink {
    command: Mutex<Option<SimulatedUserCommand>>,
}

impl EvalCommandSink {
    pub fn take(&self) -> Option<SimulatedUserCommand> {
        self.command.lock().ok()?.take()
    }

    fn submit(&self, command: SimulatedUserCommand) -> Result<(), String> {
        let mut pending = self
            .command
            .lock()
            .map_err(|_| "eval command sink is poisoned".to_string())?;
        if pending.is_some() {
            return Err("only one eval command may be submitted per turn".to_string());
        }
        *pending = Some(command);
        Ok(())
    }
}

pub struct EvalCommandTool {
    sink: Arc<EvalCommandSink>,
}

impl EvalCommandTool {
    pub fn new(sink: Arc<EvalCommandSink>) -> Self {
        Self { sink }
    }
}

impl AgentFunction for EvalCommandTool {
    fn spec(&self) -> FunctionSpec {
        FunctionSpec {
            name: "eval_command".to_string(),
            description: "Submit exactly one typed command to the evaluation VM. Use this instead of writing a command in assistant text.".to_string(),
            parameters: json!({
                "type": "object",
                "oneOf": [
                    {
                        "properties": {
                            "kind": { "const": "send_user_message" },
                            "transition": { "type": ["string", "null"] },
                            "message": { "type": "string" }
                        },
                        "required": ["kind", "transition", "message"],
                        "additionalProperties": false
                    },
                    {
                        "properties": {
                            "kind": { "const": "retry" },
                            "message": { "type": "string" },
                            "reason": { "type": "string" }
                        },
                        "required": ["kind", "message", "reason"],
                        "additionalProperties": false
                    },
                    {
                        "properties": {
                            "kind": { "const": "commit" },
                            "transition": { "type": "string" },
                            "delivery": { "enum": ["explicit", "epsilon"] },
                            "evidence": { "type": "array" },
                            "reason": { "type": "string" }
                        },
                        "required": ["kind", "transition", "delivery", "evidence", "reason"],
                        "additionalProperties": false
                    },
                    {
                        "properties": {
                            "kind": { "const": "halt" },
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
            let request: EvalCommandRequest = serde_json::from_value(args).map_err(|error| {
                lite_agent_runtime::AgentError::InvalidFunctionArguments {
                    name: "eval_command".to_string(),
                    message: error.to_string(),
                }
            })?;
            let command = request.into_command().map_err(|message| {
                lite_agent_runtime::AgentError::InvalidFunctionArguments {
                    name: "eval_command".to_string(),
                    message,
                }
            })?;
            self.sink.submit(command).map_err(|message| {
                lite_agent_runtime::AgentError::InvalidFunctionArguments {
                    name: "eval_command".to_string(),
                    message,
                }
            })?;
            Ok(FunctionExecution::Completed {
                output: json!({"accepted": true}),
            })
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum EvalCommandRequest {
    SendUserMessage {
        transition: Option<String>,
        message: String,
    },
    Retry {
        message: String,
        reason: String,
    },
    Commit {
        transition: String,
        delivery: TransitionDelivery,
        evidence: Vec<EvidenceRef>,
        reason: String,
    },
    Halt {
        reason: String,
    },
}

impl EvalCommandRequest {
    fn into_command(self) -> std::result::Result<SimulatedUserCommand, String> {
        Ok(match self {
            Self::SendUserMessage {
                transition,
                message,
            } => SimulatedUserCommand::SendUserMessage {
                transition: transition.map(TransitionId),
                message,
            },
            Self::Retry { message, reason } => SimulatedUserCommand::Retry { message, reason },
            Self::Commit {
                transition,
                delivery,
                evidence,
                reason,
            } => SimulatedUserCommand::Commit {
                transition: TransitionId(transition),
                delivery,
                evidence,
                reason,
            },
            Self::Halt { reason } => SimulatedUserCommand::Halt { reason },
        })
    }
}
