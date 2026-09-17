use crate::agent_loop::TurnAbortSignal;
use crate::error::{AgentError, Result};
use crate::model::FunctionSpec;
use canary_agent_kernel::events::Suspension;
use canary_agent_kernel::projection::ThreadProjection;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct FunctionContext {
    pub thread_id: String,
    pub metadata: Value,
    pub turn_id: String,
    pub call_id: String,
    pub projection: ThreadProjection,
    pub abort_signal: TurnAbortSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionLimits {
    pub time_budget: Duration,
    pub max_output_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDescriptor {
    pub spec: FunctionSpec,
    pub output_schema: Value,
    pub limits: FunctionLimits,
    pub recovery_policy: FunctionRecoveryPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FunctionRecoveryPolicy {
    Idempotent,
    NonIdempotent,
}

pub trait FunctionOutputResolver: Send + Sync {
    fn resolve(&self, function_name: &str, output: Value, max_output_bytes: usize)
        -> Result<Value>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DiscardResolver;

impl FunctionOutputResolver for DiscardResolver {
    fn resolve(
        &self,
        function_name: &str,
        _output: Value,
        max_output_bytes: usize,
    ) -> Result<Value> {
        Err(AgentError::FunctionOutputTooLarge {
            name: function_name.to_string(),
            max_bytes: max_output_bytes,
        })
    }
}

#[derive(Debug, Clone)]
pub enum FunctionExecution {
    Completed {
        output: Value,
    },
    SuspendedBeforeExecution {
        suspension: Suspension,
    },
    SuspendedAfterExecution {
        suspension: Suspension,
        output: Value,
    },
}

#[derive(Debug, Clone)]
pub enum SuspensionResolution {
    Approve,
    Deny { reason: String },
    UserInput { text: String },
    ExternalResult { output: Value },
}

pub trait AgentFunction: Send + Sync {
    fn spec(&self) -> FunctionSpec;
    fn output_schema(&self) -> Value;
    fn limits(&self) -> FunctionLimits;
    fn recovery_policy(&self) -> FunctionRecoveryPolicy;
    fn output_resolver(&self) -> &dyn FunctionOutputResolver;
    fn call<'a>(
        &'a self,
        args: Value,
        context: FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FunctionExecution>> + Send + 'a>>;
}

pub struct SimpleFunction<F> {
    spec: FunctionSpec,
    output_schema: Value,
    limits: FunctionLimits,
    recovery_policy: FunctionRecoveryPolicy,
    output_resolver: Arc<dyn FunctionOutputResolver>,
    handler: F,
}

impl<F> SimpleFunction<F> {
    pub fn new(
        spec: FunctionSpec,
        output_schema: Value,
        limits: FunctionLimits,
        recovery_policy: FunctionRecoveryPolicy,
        handler: F,
    ) -> Self {
        Self {
            spec,
            output_schema,
            limits,
            recovery_policy,
            output_resolver: Arc::new(DiscardResolver),
            handler,
        }
    }

    pub fn with_output_resolver<R>(mut self, resolver: R) -> Self
    where
        R: FunctionOutputResolver + 'static,
    {
        self.output_resolver = Arc::new(resolver);
        self
    }
}

impl<F, Fut> AgentFunction for SimpleFunction<F>
where
    F: Fn(Value, FunctionContext) -> Fut + Send + Sync,
    Fut: Future<Output = Result<FunctionExecution>> + Send + 'static,
{
    fn spec(&self) -> FunctionSpec {
        self.spec.clone()
    }

    fn output_schema(&self) -> Value {
        self.output_schema.clone()
    }

    fn limits(&self) -> FunctionLimits {
        self.limits
    }

    fn recovery_policy(&self) -> FunctionRecoveryPolicy {
        self.recovery_policy
    }

    fn output_resolver(&self) -> &dyn FunctionOutputResolver {
        self.output_resolver.as_ref()
    }

    fn call<'a>(
        &'a self,
        args: Value,
        context: FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FunctionExecution>> + Send + 'a>> {
        Box::pin((self.handler)(args, context))
    }
}

#[derive(Clone, Default)]
pub struct FunctionRegistry {
    functions: BTreeMap<String, Arc<dyn AgentFunction>>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<F: AgentFunction + 'static>(&mut self, function: F) {
        self.functions
            .insert(function.spec().name.clone(), Arc::new(function));
    }

    pub fn unregister(&mut self, name: &str) -> bool {
        self.functions.remove(name).is_some()
    }

    pub fn specs(&self) -> Vec<FunctionSpec> {
        self.functions
            .values()
            .map(|function| function.spec())
            .collect()
    }

    pub fn descriptors(&self) -> Vec<FunctionDescriptor> {
        self.functions
            .values()
            .map(|function| FunctionDescriptor {
                spec: function.spec(),
                output_schema: function.output_schema(),
                limits: function.limits(),
                recovery_policy: function.recovery_policy(),
            })
            .collect()
    }

    pub fn recovery_policy(&self, name: &str) -> Result<FunctionRecoveryPolicy> {
        let function = self
            .functions
            .get(name)
            .ok_or_else(|| AgentError::FunctionNotFound(name.to_string()))?;
        Ok(function.recovery_policy())
    }

    pub async fn call(
        &self,
        name: &str,
        args: Value,
        context: FunctionContext,
    ) -> Result<FunctionExecution> {
        let function = self
            .functions
            .get(name)
            .ok_or_else(|| AgentError::FunctionNotFound(name.to_string()))?;
        let limits = function.limits();
        validate_limits(name, limits)?;
        let execution = tokio::time::timeout(limits.time_budget, function.call(args, context))
            .await
            .map_err(|_| AgentError::FunctionTimeout {
                name: name.to_string(),
                timeout_ms: limits.time_budget.as_millis(),
            })??;
        enforce_output_limit(
            name,
            execution,
            limits.max_output_bytes,
            function.output_resolver(),
        )
    }
}

fn validate_limits(name: &str, limits: FunctionLimits) -> Result<()> {
    if limits.time_budget.is_zero() {
        return Err(AgentError::InvalidFunctionTimeout {
            name: name.to_string(),
        });
    }
    if limits.max_output_bytes == 0 {
        return Err(AgentError::InvalidFunctionOutputLimit {
            name: name.to_string(),
        });
    }
    Ok(())
}

fn resolve_output(
    name: &str,
    output: Value,
    max_output_bytes: usize,
    resolver: &dyn FunctionOutputResolver,
) -> Result<Value> {
    if serde_json::to_vec(&output)?.len() <= max_output_bytes {
        return Ok(output);
    }
    let resolved = resolver.resolve(name, output, max_output_bytes)?;
    if serde_json::to_vec(&resolved)?.len() > max_output_bytes {
        return Err(AgentError::FunctionOutputTooLarge {
            name: name.to_string(),
            max_bytes: max_output_bytes,
        });
    }
    Ok(resolved)
}

fn enforce_output_limit(
    name: &str,
    execution: FunctionExecution,
    max_output_bytes: usize,
    resolver: &dyn FunctionOutputResolver,
) -> Result<FunctionExecution> {
    match execution {
        FunctionExecution::Completed { output } => Ok(FunctionExecution::Completed {
            output: resolve_output(name, output, max_output_bytes, resolver)?,
        }),
        FunctionExecution::SuspendedAfterExecution { suspension, output } => {
            Ok(FunctionExecution::SuspendedAfterExecution {
                suspension,
                output: resolve_output(name, output, max_output_bytes, resolver)?,
            })
        }
        FunctionExecution::SuspendedBeforeExecution { suspension } => {
            Ok(FunctionExecution::SuspendedBeforeExecution { suspension })
        }
    }
}

#[cfg(test)]
mod tests {
    use canary_agent_kernel::projection::ThreadProjection;

    use super::{
        FunctionContext, FunctionExecution, FunctionLimits, FunctionOutputResolver,
        FunctionRecoveryPolicy, FunctionRegistry, FunctionSpec, SimpleFunction,
    };
    use crate::AgentError;
    use serde_json::json;
    use std::time::Duration;

    struct ReplaceLargeOutput;

    impl FunctionOutputResolver for ReplaceLargeOutput {
        fn resolve(
            &self,
            _function_name: &str,
            _output: serde_json::Value,
            _max_output_bytes: usize,
        ) -> crate::Result<serde_json::Value> {
            Ok(json!("ok"))
        }
    }

    #[tokio::test]
    async fn function_calls_are_stopped_at_the_declared_time_budget() {
        let mut registry = FunctionRegistry::new();
        registry.register(SimpleFunction::new(
            FunctionSpec {
                name: "never_finishes".to_string(),
                description: "Test function.".to_string(),
                parameters: json!({"type": "object"}),
            },
            json!({"type": "object"}),
            FunctionLimits {
                time_budget: Duration::from_millis(10),
                max_output_bytes: 20 * 1024 * 1024,
            },
            FunctionRecoveryPolicy::NonIdempotent,
            |_args, _context| async {
                std::future::pending::<crate::Result<FunctionExecution>>().await
            },
        ));

        let (_, abort_signal) = crate::turn_abort_pair();
        let error = registry
            .call(
                "never_finishes",
                json!({}),
                FunctionContext {
                    thread_id: "thread".to_string(),
                    metadata: serde_json::Value::Null,
                    turn_id: "turn".to_string(),
                    call_id: "call".to_string(),
                    projection: ThreadProjection::default(),
                    abort_signal,
                },
            )
            .await
            .expect_err("call should time out");

        assert!(matches!(
            error,
            AgentError::FunctionTimeout { name, timeout_ms }
                if name == "never_finishes" && timeout_ms == 10
        ));
    }

    #[tokio::test]
    async fn function_output_is_checked_against_the_declared_limit() {
        let mut registry = FunctionRegistry::new();
        registry.register(SimpleFunction::new(
            FunctionSpec {
                name: "large_output".to_string(),
                description: "Test function.".to_string(),
                parameters: json!({"type": "object"}),
            },
            json!({"type": "string"}),
            FunctionLimits {
                time_budget: Duration::from_secs(1),
                max_output_bytes: 10,
            },
            FunctionRecoveryPolicy::Idempotent,
            |_args, _context| async {
                Ok(FunctionExecution::Completed {
                    output: json!("this output is too large"),
                })
            },
        ));

        let (_, abort_signal) = crate::turn_abort_pair();
        let error = registry
            .call(
                "large_output",
                json!({}),
                FunctionContext {
                    thread_id: "thread".to_string(),
                    metadata: serde_json::Value::Null,
                    turn_id: "turn".to_string(),
                    call_id: "call".to_string(),
                    projection: ThreadProjection::default(),
                    abort_signal,
                },
            )
            .await
            .expect_err("output should exceed its limit");

        assert!(matches!(
            error,
            AgentError::FunctionOutputTooLarge { name, max_bytes }
                if name == "large_output" && max_bytes == 10
        ));
    }

    #[tokio::test]
    async fn custom_output_resolver_can_replace_large_output() {
        let mut registry = FunctionRegistry::new();
        registry.register(
            SimpleFunction::new(
                FunctionSpec {
                    name: "large_output".to_string(),
                    description: "Test function.".to_string(),
                    parameters: json!({"type": "object"}),
                },
                json!({"type": "string"}),
                FunctionLimits {
                    time_budget: Duration::from_secs(1),
                    max_output_bytes: 10,
                },
                FunctionRecoveryPolicy::Idempotent,
                |_args, _context| async {
                    Ok(FunctionExecution::Completed {
                        output: json!("this output is too large"),
                    })
                },
            )
            .with_output_resolver(ReplaceLargeOutput),
        );

        let (_, abort_signal) = crate::turn_abort_pair();
        let execution = registry
            .call(
                "large_output",
                json!({}),
                FunctionContext {
                    thread_id: "thread".to_string(),
                    metadata: serde_json::Value::Null,
                    turn_id: "turn".to_string(),
                    call_id: "call".to_string(),
                    projection: ThreadProjection::default(),
                    abort_signal,
                },
            )
            .await
            .expect("resolver should handle large output");

        let FunctionExecution::Completed { output, .. } = execution else {
            panic!("expected completed function call");
        };
        assert_eq!(output, json!("ok"));
    }
}
