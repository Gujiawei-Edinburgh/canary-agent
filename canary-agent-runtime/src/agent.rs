use crate::context::{CompactingContextBuilder, ContextBuildInput, ContextBuilder};
use crate::error::{AgentError, Result};
use crate::functions::FunctionRegistry;
use crate::metrics::{MetricsRecorder, NoopMetricsRecorder, RuntimeMetric};
use crate::model::{ModelClient, ModelRequest};
use crate::session::SessionCoordinator;
use crate::store::{ThreadContextCache, ThreadStore};
use crate::trace::{NoopTraceCollector, TraceCollector, TraceEvent, TraceEventKind};
use canary_agent_kernel::events::{Suspension, Thread, TokenUsage, TurnId, TurnItem, TurnStatus};
use canary_agent_kernel::projection::ThreadProjection;
use canary_agent_kernel::RevisionToken;
use canary_agent_revision::{
    AgentBuildRef, AgentId, AgentSpec, ModelSpec, PromptSpec, RuntimePolicySpec, ToolInterfaceSpec,
    ToolSpec,
};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::watch;

#[derive(Clone)]
pub struct AgentConfig {
    /// Allow an empty model response to leave the turn idle. Only end_turn closes it.
    pub allow_empty_response: bool,
    pub agent_id: String,
    pub turn_execution_limits: TurnExecutionLimits,
    pub system_prompt: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnExecutionLimits {
    pub max_model_iterations: usize,
    pub max_function_calls: usize,
}

impl Default for TurnExecutionLimits {
    fn default() -> Self {
        Self {
            max_model_iterations: 128,
            max_function_calls: 1024,
        }
    }
}

impl std::fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentConfig")
            .field("allow_empty_response", &self.allow_empty_response)
            .field("agent_id", &self.agent_id)
            .field("turn_execution_limits", &self.turn_execution_limits)
            .field("system_prompt", &self.system_prompt)
            .finish()
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            agent_id: "agent".to_string(),
            allow_empty_response: false,
            turn_execution_limits: TurnExecutionLimits::default(),
            system_prompt: concat!(
                "You are an agent runtime assistant. Use functions only when they are useful. ",
                "Turn items are factual append-only records. ",
                "Ask the user when required information is missing."
            )
            .to_string(),
        }
    }
}

/// Terminal outcome. Successful closure is requested through `end_turn()`.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnOutcome {
    Completed,
    Failed { error: String },
    Aborted { reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionHookEvent {
    pub phase: FunctionHookPhase,
    pub hook_fqn: String,
    pub call_id: String,
    pub name: String,
    pub message: String,
    pub metadata: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionHookPhase {
    BeforeCall,
    AfterCall,
}

pub type FunctionHookEventHandler<'a> = dyn FnMut(FunctionHookEvent) + Send + 'a;

#[derive(Debug, Clone)]
pub struct TurnAbortHandle {
    sender: watch::Sender<bool>,
}

#[derive(Debug, Clone)]
pub struct TurnAbortSignal {
    receiver: watch::Receiver<bool>,
}

pub fn turn_abort_pair() -> (TurnAbortHandle, TurnAbortSignal) {
    let (sender, receiver) = watch::channel(false);
    (TurnAbortHandle { sender }, TurnAbortSignal { receiver })
}

impl TurnAbortHandle {
    pub fn abort(&self) {
        let _ = self.sender.send(true);
    }
}

impl TurnAbortSignal {
    pub fn is_cancelled(&self) -> bool {
        *self.receiver.borrow()
    }

    pub async fn cancelled(&mut self) {
        self.wait_cancelled().await;
    }

    pub async fn wait_cancelled(&mut self) {
        if self.is_cancelled() {
            return;
        }
        while self.receiver.changed().await.is_ok() {
            if *self.receiver.borrow() {
                return;
            }
        }
        std::future::pending::<()>().await;
    }
}

#[derive(Debug, Clone)]
pub struct FunctionCallHookContext {
    pub thread_id: String,
    pub turn_id: TurnId,
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
    pub projection: ThreadProjection,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionCallHookResult {
    Completed {
        output: Value,
    },
    Suspended {
        suspension: Suspension,
        output: Value,
    },
    Failed {
        error: String,
    },
}

pub trait FunctionCallHook: Send + Sync {
    fn descriptor(&self) -> canary_agent_revision::ComponentRef;

    fn before_call<'a>(
        &'a self,
        _context: FunctionCallHookContext,
        _emit: &'a mut FunctionHookEventHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn after_call<'a>(
        &'a self,
        _context: FunctionCallHookContext,
        _result: FunctionCallHookResult,
        _emit: &'a mut FunctionHookEventHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

struct ActiveTurn {
    id: u64,
    handle: TurnAbortHandle,
}

pub(crate) struct ActiveTurnGuard {
    thread_id: String,
    id: u64,
    registry: Arc<Mutex<HashMap<String, ActiveTurn>>>,
    pub(crate) signal: TurnAbortSignal,
    pub(crate) handle: TurnAbortHandle,
}

impl Drop for ActiveTurnGuard {
    fn drop(&mut self) {
        if let Ok(mut active) = self.registry.lock() {
            if active
                .get(&self.thread_id)
                .is_some_and(|turn| turn.id == self.id)
            {
                active.remove(&self.thread_id);
            }
        }
    }
}

#[derive(Clone)]
pub struct Agent {
    pub(crate) config: AgentConfig,
    pub(crate) store: Arc<dyn ThreadStore>,
    pub(crate) model_client: Arc<dyn ModelClient>,
    pub(crate) function_registry: FunctionRegistry,
    function_call_hooks: Vec<Arc<dyn FunctionCallHook>>,
    context_builder: Arc<dyn ContextBuilder>,
    pub(crate) trace_collector: Arc<dyn TraceCollector>,
    metrics_recorder: Arc<dyn MetricsRecorder>,
    pub(crate) session_coordinator: Arc<dyn SessionCoordinator>,
    active_turns: Arc<Mutex<HashMap<String, ActiveTurn>>>,
    next_active_turn_id: Arc<AtomicU64>,
}

impl Agent {
    pub fn new(
        config: AgentConfig,
        store: Arc<dyn ThreadStore>,
        model_client: Arc<dyn ModelClient>,
        function_registry: FunctionRegistry,
        session_coordinator: Arc<dyn SessionCoordinator>,
    ) -> Self {
        Self {
            config,
            store,
            model_client,
            function_registry,
            function_call_hooks: Vec::new(),
            context_builder: Arc::new(CompactingContextBuilder::default()),
            trace_collector: Arc::new(NoopTraceCollector),
            metrics_recorder: Arc::new(NoopMetricsRecorder),
            session_coordinator,
            active_turns: Arc::new(Mutex::new(HashMap::new())),
            next_active_turn_id: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn abort(&self, thread_id: &str) -> Result<()> {
        let active = self
            .active_turns
            .lock()
            .map_err(|_| AgentError::SessionCoordinator("active turn registry poisoned".into()))?;
        let turn = active
            .get(thread_id)
            .ok_or_else(|| AgentError::TurnNotActive(thread_id.to_string()))?;
        turn.handle.abort();
        Ok(())
    }

    /// Builds a declarative snapshot from the live model client and function
    /// registry. Tool implementations are intentionally not part of it.
    pub fn spec_snapshot(&self, build: impl Into<AgentBuildRef>) -> AgentSpec {
        let model = self.model_client.model_descriptor();
        let tools = self
            .function_registry
            .descriptors()
            .into_iter()
            .map(|descriptor| {
                let name = descriptor.spec.name.clone();
                (
                    name.clone(),
                    ToolSpec {
                        interface: ToolInterfaceSpec {
                            name,
                            description: descriptor.spec.description,
                            parameters: descriptor.spec.parameters,
                            output_schema: descriptor.output_schema,
                        },
                        configuration: serde_json::json!({}),
                    },
                )
            })
            .collect();

        AgentSpec {
            agent_id: AgentId::from(self.config.agent_id.as_str()),
            model: ModelSpec {
                fqn: model.fqn,
                settings: model.settings,
            },
            prompts: PromptSpec {
                system: self.config.system_prompt.clone(),
                templates: BTreeMap::new(),
            },
            tools,
            runtime: RuntimePolicySpec {
                allow_empty_response: self.config.allow_empty_response,
                context_builder: self.context_builder.descriptor(),
                hooks: self
                    .function_call_hooks
                    .iter()
                    .map(|hook| hook.descriptor())
                    .collect(),
                turn_execution_limits: canary_agent_revision::TurnExecutionLimitsSpec {
                    max_model_iterations: self.config.turn_execution_limits.max_model_iterations,
                    max_function_calls: self.config.turn_execution_limits.max_function_calls,
                },
            },
            build: build.into(),
        }
    }

    pub(crate) fn register_active_turn(&self, thread_id: &str) -> Result<ActiveTurnGuard> {
        let (handle, signal) = turn_abort_pair();
        let id = self.next_active_turn_id.fetch_add(1, Ordering::Relaxed);
        let mut active = self
            .active_turns
            .lock()
            .map_err(|_| AgentError::SessionCoordinator("active turn registry poisoned".into()))?;
        if active.contains_key(thread_id) {
            return Err(AgentError::TurnAlreadyActive(thread_id.to_string()));
        }
        active.insert(
            thread_id.to_string(),
            ActiveTurn {
                id,
                handle: handle.clone(),
            },
        );
        Ok(ActiveTurnGuard {
            thread_id: thread_id.to_string(),
            id,
            registry: self.active_turns.clone(),
            signal,
            handle,
        })
    }

    pub fn with_function_call_hook<H>(mut self, hook: H) -> Self
    where
        H: FunctionCallHook + 'static,
    {
        self.function_call_hooks.push(Arc::new(hook));
        self
    }

    pub fn with_function<F>(mut self, function: F) -> Self
    where
        F: crate::functions::AgentFunction + 'static,
    {
        self.function_registry.register(function);
        self
    }

    pub fn without_function(mut self, name: &str) -> Self {
        self.function_registry.unregister(name);
        self
    }

    pub fn with_function_call_hooks(mut self, hooks: Vec<Arc<dyn FunctionCallHook>>) -> Self {
        self.function_call_hooks.extend(hooks);
        self
    }

    pub fn with_context_builder<C>(mut self, context_builder: C) -> Self
    where
        C: ContextBuilder + 'static,
    {
        self.context_builder = Arc::new(context_builder);
        self
    }

    pub fn with_trace_collector<C>(mut self, trace_collector: C) -> Self
    where
        C: TraceCollector + 'static,
    {
        self.trace_collector = Arc::new(trace_collector);
        self
    }

    pub fn with_shared_trace_collector(mut self, trace_collector: Arc<dyn TraceCollector>) -> Self {
        self.trace_collector = trace_collector;
        self
    }

    pub fn with_metrics_recorder<M>(mut self, metrics_recorder: M) -> Self
    where
        M: MetricsRecorder + 'static,
    {
        self.metrics_recorder = Arc::new(metrics_recorder);
        self
    }

    pub(crate) fn record_metric(&self, metric: RuntimeMetric) {
        self.metrics_recorder.record(metric);
    }

    pub(crate) fn apply_turn_token_usage(thread: &mut Thread, usage: TokenUsage) {
        if !usage.is_zero() {
            thread.token_usage.add_assign(usage);
        }
    }

    pub(crate) fn record_trace(
        &self,
        sequence: &mut u64,
        thread_id: &str,
        turn_id: &str,
        kind: TraceEventKind,
    ) {
        *sequence = sequence.saturating_add(1);
        self.trace_collector.record(TraceEvent {
            thread_id: thread_id.to_string(),
            turn_id: turn_id.to_string(),
            sequence: *sequence,
            occurred_at: canary_agent_kernel::now_timestamp(),
            kind,
        });
    }

    pub(crate) async fn run_before_function_call_hooks(
        &self,
        context: &FunctionCallHookContext,
        on_event: &mut FunctionHookEventHandler<'_>,
    ) -> (Vec<Arc<dyn FunctionCallHook>>, Result<()>) {
        let mut entered_hooks = Vec::new();
        for hook in &self.function_call_hooks {
            if let Err(error) = hook.before_call((*context).clone(), on_event).await {
                return (entered_hooks, Err(error));
            }
            entered_hooks.push(hook.clone());
        }
        (entered_hooks, Ok(()))
    }

    pub(crate) async fn run_after_function_call_hooks(
        &self,
        hooks: &[Arc<dyn FunctionCallHook>],
        context: &FunctionCallHookContext,
        result: FunctionCallHookResult,
        on_event: &mut FunctionHookEventHandler<'_>,
    ) {
        for hook in hooks.iter().rev() {
            if let Err(error) = hook
                .after_call((*context).clone(), result.clone(), on_event)
                .await
            {
                tracing::warn!(error = %error, "function call post-hook failed");
                on_event(FunctionHookEvent {
                    phase: FunctionHookPhase::AfterCall,
                    hook_fqn: hook.descriptor().fqn,
                    call_id: context.call_id.clone(),
                    name: context.name.clone(),
                    message: "post-hook failed".to_string(),
                    metadata: serde_json::json!({
                        "call_id": context.call_id,
                        "name": context.name,
                        "error": error.to_string(),
                    }),
                });
            }
        }
    }

    pub(crate) async fn commit_thread(
        &self,
        thread: Thread,
        lease_fence: &crate::session::LeaseFence,
    ) -> Result<Thread> {
        self.store.compare_and_commit(thread, lease_fence).await
    }

    pub(crate) async fn model_request_from_projection(
        &self,
        thread_id: &str,
        thread_revision: &RevisionToken,
        projection: ThreadProjection,
        cached_context: Option<&ThreadContextCache>,
    ) -> Result<ModelRequest> {
        let context = self
            .context_builder
            .build(ContextBuildInput {
                thread_id,
                thread_revision,
                projection: &projection,
                system_prompt: &self.config.system_prompt,
                cached_context,
            })
            .await?;
        if let Some(cache) = context.cache {
            self.store.save_context_cache(cache).await?;
        }
        Ok(ModelRequest {
            messages: context.messages,
            functions: self.function_registry.specs(),
        })
    }

    pub(crate) fn push_turn_items(
        thread: &mut Thread,
        turn_id: &str,
        items: Vec<TurnItem>,
    ) -> Result<()> {
        let turn = thread
            .turn_mut(turn_id)
            .ok_or_else(|| AgentError::TurnNotFound(turn_id.to_string()))?;
        for item in items {
            turn.push_item(item);
        }
        Ok(())
    }

    pub(crate) fn set_turn_status(
        thread: &mut Thread,
        turn_id: &str,
        status: TurnStatus,
    ) -> Result<()> {
        let turn = thread
            .turn_mut(turn_id)
            .ok_or_else(|| AgentError::TurnNotFound(turn_id.to_string()))?;
        turn.set_status(status);
        Ok(())
    }
}
