use crate::context::{CompactingContextBuilder, ContextBuildInput, ContextBuilder};
use crate::error::{AgentError, Result};
use crate::functions::{
    FunctionCallExecution, FunctionContext, FunctionRecoveryPolicy, FunctionRegistry,
    RuntimeEffect, SuspensionResolution,
};
use crate::metrics::{
    FunctionCallOutcome, FunctionCallSkipReason, MetricStatus, MetricsRecorder,
    NoopMetricsRecorder, RuntimeMetric,
};
use crate::model::{ModelClient, ModelFunctionCall, ModelRequest, ModelResponse, ModelStreamEvent};
use crate::session::{SessionCoordinator, SessionLease};
use crate::store::{ThreadContextCache, ThreadStore};
use crate::trace::{
    NoopTraceCollector, TraceCollector, TraceEvent, TraceEventKind, TraceTurnStatus,
};
use canary_agent_kernel::events::{
    new_id, Suspension, SuspensionKind, Thread, TokenUsage, ToolResult, Turn, TurnId, TurnItem,
    TurnItemKind, TurnItemSource, TurnStatus,
};
use canary_agent_kernel::projection::ThreadProjection;
use canary_agent_kernel::RevisionToken;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::watch;

#[derive(Clone)]
pub struct AgentConfig {
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
            .field("turn_execution_limits", &self.turn_execution_limits)
            .field("system_prompt", &self.system_prompt)
            .finish()
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            turn_execution_limits: TurnExecutionLimits::default(),
            system_prompt: concat!(
                "You are an agent runtime assistant. Use functions only when they are useful. ",
                "Thread goal is explicit durable state. Turn items are factual append-only records. ",
                "Ask the user when required information is missing."
            )
            .to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnOutcome {
    AssistantMessage { text: String },
    Suspended { suspension: Suspension },
    Failed { error: String },
    Aborted { reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnStreamEvent {
    State(TurnStateEvent),
    Model(TurnModelEvent),
    Runtime(RuntimeEvent),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnStateEvent {
    TurnStarted {
        thread_id: String,
        turn_id: TurnId,
    },
    FunctionCallsRequested {
        calls: Vec<crate::model::ModelFunctionCall>,
    },
    FunctionStarted {
        call_id: String,
        name: String,
    },
    FunctionCompleted {
        call_id: String,
        name: String,
    },
    FunctionFailed {
        call_id: String,
        name: String,
        error: String,
    },
    Suspended {
        suspension: Suspension,
    },
    TurnFinished {
        outcome: TurnOutcome,
    },
    TurnFailed {
        error: String,
    },
    TurnAborted {
        reason: String,
    },
    TurnTokenUsage {
        usage: TokenUsage,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnModelEvent {
    RequestStarted { iteration: usize },
    AssistantMessage { text: String },
    AssistantDelta { text: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeEvent {
    pub source: String,
    pub message: String,
    pub metadata: Value,
}

pub type TurnEventHandler<'a> = dyn FnMut(TurnStreamEvent) + Send + 'a;

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
    fn before_call<'a>(
        &'a self,
        _context: FunctionCallHookContext,
        _emit: &'a mut TurnEventHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn after_call<'a>(
        &'a self,
        _context: FunctionCallHookContext,
        _result: FunctionCallHookResult,
        _emit: &'a mut TurnEventHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

#[derive(Debug, Clone)]
struct Session {
    _thread_id: String,
    active_turn_id: TurnId,
    projection: ThreadProjection,
}

struct ActiveTurn {
    id: u64,
    handle: TurnAbortHandle,
}

struct ActiveTurnGuard {
    thread_id: String,
    id: u64,
    registry: Arc<Mutex<HashMap<String, ActiveTurn>>>,
    signal: TurnAbortSignal,
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

impl Session {
    fn from_thread(thread: &Thread, active_turn_id: TurnId) -> Self {
        Self {
            _thread_id: thread.id.clone(),
            active_turn_id,
            projection: ThreadProjection::from_thread(thread),
        }
    }
}

#[derive(Clone)]
pub struct Agent {
    config: AgentConfig,
    store: Arc<dyn ThreadStore>,
    model_client: Arc<dyn ModelClient>,
    function_registry: FunctionRegistry,
    function_call_hooks: Vec<Arc<dyn FunctionCallHook>>,
    context_builder: Arc<dyn ContextBuilder>,
    trace_collector: Arc<dyn TraceCollector>,
    metrics_recorder: Arc<dyn MetricsRecorder>,
    session_coordinator: Arc<dyn SessionCoordinator>,
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

    fn register_active_turn(&self, thread_id: &str) -> Result<ActiveTurnGuard> {
        let (handle, signal) = turn_abort_pair();
        let id = self.next_active_turn_id.fetch_add(1, Ordering::Relaxed);
        let mut active = self
            .active_turns
            .lock()
            .map_err(|_| AgentError::SessionCoordinator("active turn registry poisoned".into()))?;
        if active.contains_key(thread_id) {
            return Err(AgentError::TurnAlreadyActive(thread_id.to_string()));
        }
        active.insert(thread_id.to_string(), ActiveTurn { id, handle });
        Ok(ActiveTurnGuard {
            thread_id: thread_id.to_string(),
            id,
            registry: self.active_turns.clone(),
            signal,
        })
    }

    pub fn with_function_call_hook<H>(mut self, hook: H) -> Self
    where
        H: FunctionCallHook + 'static,
    {
        self.function_call_hooks.push(Arc::new(hook));
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

    fn record_metric(&self, metric: RuntimeMetric) {
        self.metrics_recorder.record(metric);
    }

    /// Starts a turn and applies `metadata` when the thread does not exist yet.
    /// Existing thread metadata is preserved.
    pub async fn run_turn<'a, F>(
        &self,
        thread_id: &str,
        user_text: impl Into<String>,
        metadata: Value,
        on_event: F,
    ) -> Result<TurnOutcome>
    where
        F: FnMut(TurnStreamEvent) + Send + 'a,
    {
        let active_turn = self.register_active_turn(thread_id)?;
        let lease = self.session_coordinator.acquire(thread_id).await?;
        self.run_turn_internal(
            thread_id,
            Some(user_text.into()),
            Some(metadata),
            active_turn.signal.clone(),
            on_event,
            0,
            &lease,
        )
        .await
    }

    /// Recover a turn left running by a previous process after a crash.
    ///
    /// Calls that were never started are executed normally. Calls with a
    /// durable start marker but no terminal output are retried only when the
    /// registered function is idempotent; otherwise the turn is suspended for
    /// external reconciliation.
    pub async fn recover_turn<'a, F>(&self, thread_id: &str, on_event: F) -> Result<TurnOutcome>
    where
        F: FnMut(TurnStreamEvent) + Send + 'a,
    {
        let active_turn = self.register_active_turn(thread_id)?;
        let lease = self.session_coordinator.acquire(thread_id).await?;
        self.run_turn_internal(
            thread_id,
            None,
            None,
            active_turn.signal.clone(),
            on_event,
            0,
            &lease,
        )
        .await
    }

    /// Resume the existing turn associated with a suspension.
    ///
    /// The caller must update any external authorization state before calling
    /// this method. The original function call is executed in the suspended
    /// turn; no user-input item or new turn is created.
    pub async fn resume_suspended_turn<'a, F>(
        &self,
        thread_id: &str,
        suspension_id: &str,
        resolution: SuspensionResolution,
        on_event: F,
    ) -> Result<TurnOutcome>
    where
        F: FnMut(TurnStreamEvent) + Send + 'a,
    {
        let active_turn = self.register_active_turn(thread_id)?;
        let lease = self.session_coordinator.acquire(thread_id).await?;
        let mut abort_signal = active_turn.signal.clone();
        let mut on_event = on_event;
        let mut suspended_outcome = None;
        let should_continue = {
            let mut thread = self.store.load(thread_id).await?;
            let projection = ThreadProjection::from_thread(&thread);
            let pending = projection
                .pending_suspension
                .clone()
                .ok_or_else(|| AgentError::TurnNotFound(suspension_id.to_string()))?;
            if pending.suspension.id != suspension_id {
                return Err(AgentError::TurnNotFound(suspension_id.to_string()));
            }
            let deferred = pending
                .suspension
                .payload
                .get("deferred")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if let SuspensionResolution::UserInput { text } = resolution.clone() {
                if deferred {
                    return Err(AgentError::Function {
                        name: "resume_suspended_turn".to_string(),
                        message: "a pre-execution suspension requires approval or denial"
                            .to_string(),
                    });
                }
                Self::push_turn_items(
                    &mut thread,
                    &pending.turn_id,
                    vec![TurnItem::new(
                        TurnItemSource::User,
                        TurnItemKind::UserInput {
                            text,
                            response_to: Some(suspension_id.to_string()),
                        },
                    )],
                )?;
                Self::set_turn_status(&mut thread, &pending.turn_id, TurnStatus::Running)?;
                self.commit_thread(thread, lease.fence()).await?;
                true
            } else {
                let call_id = pending
                    .suspension
                    .payload
                    .get("call_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| AgentError::Function {
                        name: "resume_suspended_turn".to_string(),
                        message: "suspension does not identify a resumable call".to_string(),
                    })?;
                let call = thread
                    .turns
                    .iter()
                    .find(|turn| turn.id == pending.turn_id)
                    .and_then(|turn| {
                        turn.items.iter().rev().find_map(|item| match &item.kind {
                            TurnItemKind::ModelResponse { function_calls, .. } => function_calls
                                .iter()
                                .find(|call| call.call_id == call_id)
                                .cloned(),
                            _ => None,
                        })
                    })
                    .ok_or_else(|| AgentError::Function {
                        name: "resume_suspended_turn".to_string(),
                        message: "suspended function call was not found".to_string(),
                    })?;
                if !deferred {
                    return Err(AgentError::Function {
                        name: "resume_suspended_turn".to_string(),
                        message: "approval or denial can only resume a pre-execution suspension"
                            .to_string(),
                    });
                }
                if let SuspensionResolution::Deny { reason } = resolution.clone() {
                    let error_text = format!("execution denied by user: {reason}");
                    Self::push_turn_items(
                        &mut thread,
                        &pending.turn_id,
                        vec![TurnItem::new(
                            TurnItemSource::Tool,
                            TurnItemKind::ToolOutput {
                                call_id: call.call_id.clone(),
                                name: call.name.clone(),
                                result: ToolResult::Error {
                                    error: error_text.clone(),
                                },
                            },
                        )],
                    )?;
                    Self::set_turn_status(&mut thread, &pending.turn_id, TurnStatus::Running)?;
                    self.commit_thread(thread, lease.fence()).await?;
                    on_event(TurnStreamEvent::State(TurnStateEvent::FunctionFailed {
                        call_id: call.call_id,
                        name: call.name,
                        error: error_text,
                    }));
                    true
                } else if matches!(resolution, SuspensionResolution::Approve) {
                    Self::push_turn_items(
                        &mut thread,
                        &pending.turn_id,
                        vec![TurnItem::new(
                            TurnItemSource::Runtime,
                            TurnItemKind::FunctionCallStarted {
                                call_id: call.call_id.clone(),
                                name: call.name.clone(),
                                attempt: pending
                                    .suspension
                                    .payload
                                    .get("attempt")
                                    .and_then(Value::as_u64)
                                    .unwrap_or(1) as u32
                                    + 1,
                            },
                        )],
                    )?;
                    thread = self.commit_thread(thread, lease.fence()).await?;
                    on_event(TurnStreamEvent::State(TurnStateEvent::FunctionStarted {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                    }));
                    let context = FunctionContext {
                        thread_id: thread.id.clone(),
                        metadata: thread.metadata.clone(),
                        turn_id: pending.turn_id.clone(),
                        call_id: call.call_id.clone(),
                        projection: projection.clone(),
                        abort_signal: abort_signal.clone(),
                    };
                    let execution =
                        self.function_registry
                            .call(&call.name, call.arguments.clone(), context);
                    let execution = self
                        .await_step_or_abort(
                            execution,
                            &mut abort_signal,
                            &mut thread,
                            thread_id,
                            &pending.turn_id,
                            "resumed function call",
                        )
                        .await?;
                    let Some(execution) = execution else {
                        Self::push_turn_items(
                            &mut thread,
                            &pending.turn_id,
                            vec![TurnItem::new(
                                TurnItemSource::Tool,
                                TurnItemKind::ToolOutput {
                                    call_id: call.call_id.clone(),
                                    name: call.name.clone(),
                                    result: ToolResult::Aborted {
                                        reason: "function execution aborted before a result was recorded"
                                            .to_string(),
                                    },
                                },
                            )],
                        )?;
                        self.commit_thread(thread, lease.fence()).await?;
                        return Ok(TurnOutcome::Aborted {
                            reason: "turn aborted by caller".to_string(),
                        });
                    };
                    match execution {
                        Ok(FunctionCallExecution::Completed { output, effects }) => {
                            let mut items = Self::apply_runtime_effects(&thread, effects);
                            items.push(TurnItem::new(
                                TurnItemSource::Tool,
                                TurnItemKind::ToolOutput {
                                    call_id: call.call_id.clone(),
                                    name: call.name.clone(),
                                    result: ToolResult::Success {
                                        output: output.clone(),
                                    },
                                },
                            ));
                            Self::push_turn_items(&mut thread, &pending.turn_id, items)?;
                            Self::set_turn_status(
                                &mut thread,
                                &pending.turn_id,
                                TurnStatus::Running,
                            )?;
                            self.commit_thread(thread, lease.fence()).await?;
                            on_event(TurnStreamEvent::State(TurnStateEvent::FunctionCompleted {
                                call_id: call.call_id,
                                name: call.name,
                            }));
                            true
                        }
                        Ok(FunctionCallExecution::SuspendedBeforeExecution {
                            suspension, ..
                        })
                        | Ok(FunctionCallExecution::SuspendedAfterExecution {
                            suspension, ..
                        }) => {
                            suspended_outcome = Some(TurnOutcome::Suspended { suspension });
                            false
                        }
                        Err(error) => {
                            let error_text = error.to_string();
                            Self::push_turn_items(
                                &mut thread,
                                &pending.turn_id,
                                vec![TurnItem::new(
                                    TurnItemSource::Tool,
                                    TurnItemKind::ToolOutput {
                                        call_id: call.call_id.clone(),
                                        name: call.name.clone(),
                                        result: ToolResult::Error {
                                            error: error_text.clone(),
                                        },
                                    },
                                )],
                            )?;
                            Self::set_turn_status(
                                &mut thread,
                                &pending.turn_id,
                                TurnStatus::Running,
                            )?;
                            self.commit_thread(thread, lease.fence()).await?;
                            on_event(TurnStreamEvent::State(TurnStateEvent::FunctionFailed {
                                call_id: call.call_id,
                                name: call.name,
                                error: error_text,
                            }));
                            true
                        }
                    }
                } else {
                    return Err(AgentError::Function {
                        name: "resume_suspended_turn".to_string(),
                        message: "unsupported suspension resolution".to_string(),
                    });
                }
            }
        };
        if let Some(outcome) = suspended_outcome {
            return Ok(outcome);
        }
        if should_continue {
            return self
                .run_turn_internal(thread_id, None, None, abort_signal, on_event, 1, &lease)
                .await;
        }
        Ok(TurnOutcome::Failed {
            error: "suspended turn could not be resumed".to_string(),
        })
    }

    async fn run_turn_internal<'a, F>(
        &self,
        thread_id: &str,
        user_text: Option<String>,
        initial_metadata: Option<Value>,
        mut abort_signal: TurnAbortSignal,
        mut on_event: F,
        mut trace_sequence: u64,
        lease: &SessionLease,
    ) -> Result<TurnOutcome>
    where
        F: FnMut(TurnStreamEvent) + Send + 'a,
    {
        tracing::debug!(thread_id, "turn started with session lease");
        let turn_started = Instant::now();

        let mut thread = match self.store.load(thread_id).await {
            Ok(thread) => thread,
            Err(AgentError::ThreadNotFound(_)) => {
                let mut thread = Thread::new(thread_id);
                if let Some(metadata) = initial_metadata {
                    thread.metadata = metadata;
                }
                thread
            }
            Err(error) => return Err(error),
        };
        let recovering = user_text.is_none();
        let (turn_id, trace_user_text) = if let Some(user_text) = user_text {
            if let Some(pending) = ThreadProjection::from_thread(&thread).pending_suspension {
                return Err(AgentError::SuspendedTurn {
                    thread_id: thread_id.to_string(),
                    suspension_id: pending.suspension.id,
                });
            }
            let mut turn = Turn::new();
            let turn_id = turn.id.clone();
            turn.push_item(TurnItem::new(
                TurnItemSource::User,
                TurnItemKind::UserInput {
                    text: user_text.clone(),
                    response_to: None,
                },
            ));
            thread.turns.push(turn);
            thread = self.commit_thread(thread, lease.fence()).await?;
            (turn_id, Some(user_text))
        } else {
            let projection = ThreadProjection::from_thread(&thread);
            if let Some(pending) = projection.pending_suspension {
                return Err(AgentError::SuspendedTurn {
                    thread_id: thread_id.to_string(),
                    suspension_id: pending.suspension.id,
                });
            }
            let turn_id = projection
                .active_turn_id
                .ok_or_else(|| AgentError::TurnNotFound(thread_id.to_string()))?;
            (turn_id, None)
        };
        tracing::info!(thread_id, turn_id, "turn started");
        if let Some(trace_user_text) = trace_user_text {
            self.record_trace(
                &mut trace_sequence,
                thread_id,
                &turn_id,
                TraceEventKind::UserInput {
                    text: trace_user_text,
                    response_to: None,
                },
            );
        }
        on_event(TurnStreamEvent::State(TurnStateEvent::TurnStarted {
            thread_id: thread_id.to_string(),
            turn_id: turn_id.clone(),
        }));
        let mut turn_token_usage = TokenUsage::default();
        let mut function_call_count = 0;
        let mut first_token_recorded = false;
        let limits = self.config.turn_execution_limits;

        let outcome = 'turn_loop: {
            if recovering {
                if let Some(outcome) = self
                    .recover_pending_function_calls(
                        &mut thread,
                        &turn_id,
                        &mut abort_signal,
                        lease,
                        &mut on_event,
                        &mut trace_sequence,
                    )
                    .await?
                {
                    break 'turn_loop outcome;
                }
            }
            for iteration in 0..limits.max_model_iterations {
                let session = Session::from_thread(&thread, turn_id.clone());
                let cached_context = self.store.load_context_cache(thread_id).await?;
                let request = self
                    .model_request_from_projection(
                        thread_id,
                        &thread.revision,
                        session.projection.clone(),
                        cached_context.as_ref(),
                    )
                    .await?;
                on_event(TurnStreamEvent::Model(TurnModelEvent::RequestStarted {
                    iteration,
                }));
                let mut model_event_handler = |event| match event {
                    ModelStreamEvent::AssistantDelta { text } => {
                        if !first_token_recorded {
                            first_token_recorded = true;
                            self.record_metric(RuntimeMetric::TimeToFirstToken {
                                duration: turn_started.elapsed(),
                            });
                        }
                        on_event(TurnStreamEvent::Model(TurnModelEvent::AssistantDelta {
                            text,
                        }));
                    }
                    ModelStreamEvent::TokenUsage { usage } => {
                        turn_token_usage.add_assign(usage);
                    }
                };
                let model_call = self
                    .model_client
                    .stream_complete(request, &mut model_event_handler);
                let model_started = Instant::now();
                let response = self
                    .await_step_or_abort(
                        model_call,
                        &mut abort_signal,
                        &mut thread,
                        thread_id,
                        &session.active_turn_id,
                        "model request",
                    )
                    .await?;
                self.record_metric(RuntimeMetric::ModelRequestFinished {
                    status: match &response {
                        None => MetricStatus::Aborted,
                        Some(Ok(_)) => MetricStatus::Completed,
                        Some(Err(_)) => MetricStatus::Failed,
                    },
                    duration: model_started.elapsed(),
                });
                let Some(response) = response else {
                    break 'turn_loop TurnOutcome::Aborted {
                        reason: "turn aborted by caller".to_string(),
                    };
                };
                let response = match response {
                    Ok(response) => response,
                    Err(error) => {
                        let error = error.to_string();
                        tracing::error!(error, "turn failed during model request");
                        self.fail_turn(&mut thread, &session.active_turn_id, error.clone())?;
                        break 'turn_loop TurnOutcome::Failed { error };
                    }
                };

                let response = match response {
                    ModelResponse::AssistantMessage { text } => ModelResponse::Assistant {
                        text: Some(text),
                        function_calls: Vec::new(),
                    },
                    ModelResponse::FunctionCalls { calls } => ModelResponse::Assistant {
                        text: None,
                        function_calls: calls,
                    },
                    response => response,
                };
                match response {
                    ModelResponse::Assistant {
                        text,
                        function_calls,
                    } => {
                        if text.is_none() && function_calls.is_empty() {
                            let error = "model returned neither assistant text nor function calls"
                                .to_string();
                            tracing::error!(error, "empty model response");
                            self.fail_turn(&mut thread, &session.active_turn_id, error.clone())?;
                            break 'turn_loop TurnOutcome::Failed { error };
                        }
                        if let Some(text) = &text {
                            on_event(TurnStreamEvent::Model(TurnModelEvent::AssistantMessage {
                                text: text.clone(),
                            }));
                        }
                        let trace_model_response = TraceEventKind::ModelResponse {
                            text: text.clone(),
                            function_calls: function_calls.clone(),
                        };
                        Self::push_turn_items(
                            &mut thread,
                            &session.active_turn_id,
                            vec![TurnItem::new(
                                TurnItemSource::Model,
                                TurnItemKind::ModelResponse {
                                    text: text.clone(),
                                    function_calls: function_calls.clone(),
                                },
                            )],
                        )?;
                        if function_calls.is_empty() {
                            let text = text.unwrap_or_default();
                            Self::set_turn_status(
                                &mut thread,
                                &session.active_turn_id,
                                TurnStatus::Completed,
                            )?;
                            thread = self.commit_thread(thread, lease.fence()).await?;
                            self.record_trace(
                                &mut trace_sequence,
                                thread_id,
                                &session.active_turn_id,
                                trace_model_response,
                            );
                            break 'turn_loop TurnOutcome::AssistantMessage { text };
                        }
                        let calls = function_calls;
                        if calls.is_empty() {
                            let error = "model returned an empty function call list".to_string();
                            tracing::warn!(error, "empty function call list");
                            self.fail_turn(&mut thread, &session.active_turn_id, error.clone())?;
                            break 'turn_loop TurnOutcome::Failed { error };
                        }

                        on_event(TurnStreamEvent::State(
                            TurnStateEvent::FunctionCallsRequested {
                                calls: calls.clone(),
                            },
                        ));
                        thread = self.commit_thread(thread, lease.fence()).await?;
                        self.record_trace(
                            &mut trace_sequence,
                            thread_id,
                            &session.active_turn_id,
                            trace_model_response,
                        );

                        let remaining_function_calls = limits
                            .max_function_calls
                            .saturating_sub(function_call_count);
                        let admitted_call_count = calls.len().min(remaining_function_calls);
                        function_call_count += admitted_call_count;

                        for (call_index, call) in calls.iter().take(admitted_call_count).enumerate()
                        {
                            let call_id = call.call_id.clone();
                            let name = call.name.clone();
                            on_event(TurnStreamEvent::State(TurnStateEvent::FunctionStarted {
                                call_id: call_id.clone(),
                                name: name.clone(),
                            }));
                            self.record_trace(
                                &mut trace_sequence,
                                thread_id,
                                &turn_id,
                                TraceEventKind::FunctionCall {
                                    call_id: call_id.clone(),
                                    name: name.clone(),
                                    arguments: call.arguments.clone(),
                                },
                            );
                            let mut hook_context = FunctionCallHookContext {
                                thread_id: thread.id.clone(),
                                turn_id: turn_id.clone(),
                                call_id: call_id.clone(),
                                name: name.clone(),
                                arguments: call.arguments.clone(),
                                projection: ThreadProjection::from_thread(&thread),
                            };
                            let (entered_hooks, pre_hook_result) = self
                                .run_before_function_call_hooks(&hook_context, &mut on_event)
                                .await;
                            let function_started = Instant::now();
                            let execution = match pre_hook_result {
                                Ok(()) => {
                                    Self::push_turn_items(
                                        &mut thread,
                                        &turn_id,
                                        vec![TurnItem::new(
                                            TurnItemSource::Runtime,
                                            TurnItemKind::FunctionCallStarted {
                                                call_id: call_id.clone(),
                                                name: name.clone(),
                                                attempt: 1,
                                            },
                                        )],
                                    )?;
                                    thread = self.commit_thread(thread, lease.fence()).await?;
                                    hook_context.projection =
                                        ThreadProjection::from_thread(&thread);
                                    let context = FunctionContext {
                                        thread_id: thread.id.clone(),
                                        metadata: thread.metadata.clone(),
                                        turn_id: turn_id.clone(),
                                        call_id: call_id.clone(),
                                        projection: hook_context.projection.clone(),
                                        abort_signal: abort_signal.clone(),
                                    };
                                    let function_call = self.function_registry.call(
                                        &call.name,
                                        call.arguments.clone(),
                                        context,
                                    );
                                    self.await_step_or_abort(
                                        function_call,
                                        &mut abort_signal,
                                        &mut thread,
                                        thread_id,
                                        &turn_id,
                                        "function call",
                                    )
                                    .await?
                                }
                                Err(error) => Some(Err(error)),
                            };

                            let Some(execution) = execution else {
                                self.record_metric(RuntimeMetric::FunctionCallFinished {
                                    name: name.clone(),
                                    outcome: FunctionCallOutcome::Aborted,
                                    duration: function_started.elapsed(),
                                });
                                for skipped_call in calls.iter().skip(call_index + 1) {
                                    self.record_metric(RuntimeMetric::FunctionCallSkipped {
                                        name: skipped_call.name.clone(),
                                        reason: FunctionCallSkipReason::TurnAborted,
                                    });
                                }
                                Self::append_aborted_function_outputs(
                                    &mut thread,
                                    &turn_id,
                                    &calls,
                                    call_index,
                                )?;
                                break 'turn_loop TurnOutcome::Aborted {
                                    reason: "turn aborted by caller".to_string(),
                                };
                            };
                            self.record_metric(RuntimeMetric::FunctionCallFinished {
                                name: name.clone(),
                                outcome: match &execution {
                                    Ok(FunctionCallExecution::Completed { .. }) => {
                                        FunctionCallOutcome::Completed
                                    }
                                    Ok(FunctionCallExecution::SuspendedBeforeExecution {
                                        ..
                                    })
                                    | Ok(FunctionCallExecution::SuspendedAfterExecution {
                                        ..
                                    }) => FunctionCallOutcome::Suspended,
                                    Err(_) => FunctionCallOutcome::Failed,
                                },
                                duration: function_started.elapsed(),
                            });
                            match execution {
                                Ok(FunctionCallExecution::Completed { output, effects }) => {
                                    let update_items =
                                        Self::apply_runtime_effects(&thread, effects);
                                    let hook_result = FunctionCallHookResult::Completed {
                                        output: output.clone(),
                                    };
                                    let trace_tool_output = TraceEventKind::ToolOutput {
                                        call_id: call_id.clone(),
                                        name: name.clone(),
                                        result: ToolResult::Success {
                                            output: output.clone(),
                                        },
                                    };
                                    let mut func_items = update_items;
                                    func_items.push(TurnItem::new(
                                        TurnItemSource::Tool,
                                        TurnItemKind::ToolOutput {
                                            call_id: call_id.clone(),
                                            name: name.clone(),
                                            result: ToolResult::Success { output },
                                        },
                                    ));
                                    Self::push_turn_items(&mut thread, &turn_id, func_items)?;
                                    thread = self.commit_thread(thread, lease.fence()).await?;
                                    self.record_trace(
                                        &mut trace_sequence,
                                        thread_id,
                                        &turn_id,
                                        trace_tool_output,
                                    );
                                    hook_context.projection =
                                        ThreadProjection::from_thread(&thread);
                                    self.run_after_function_call_hooks(
                                        &entered_hooks,
                                        &hook_context,
                                        hook_result,
                                        &mut on_event,
                                    )
                                    .await;
                                    on_event(TurnStreamEvent::State(
                                        TurnStateEvent::FunctionCompleted { call_id, name },
                                    ));
                                }
                                Ok(FunctionCallExecution::SuspendedBeforeExecution {
                                    suspension,
                                    effects,
                                }) => {
                                    let mut func_items =
                                        Self::apply_runtime_effects(&thread, effects);
                                    func_items.push(TurnItem::new(
                                        TurnItemSource::Runtime,
                                        TurnItemKind::SuspensionCreated {
                                            suspension: suspension.clone(),
                                        },
                                    ));
                                    let skipped_calls = calls
                                        .iter()
                                        .skip(call_index + 1)
                                        .map(|skipped_call| {
                                            let error = "function not executed because a previous function suspended the turn";
                                            func_items.push(TurnItem::new(
                                                TurnItemSource::Tool,
                                                TurnItemKind::ToolOutput {
                                                    call_id: skipped_call.call_id.clone(),
                                                    name: skipped_call.name.clone(),
                                                    result: ToolResult::Error {
                                                        error: error.to_string(),
                                                    },
                                                },
                                            ));
                                            (
                                                skipped_call.call_id.clone(),
                                                skipped_call.name.clone(),
                                                error.to_string(),
                                            )
                                        })
                                        .collect::<Vec<_>>();
                                    for (_, name, _) in &skipped_calls {
                                        self.record_metric(RuntimeMetric::FunctionCallSkipped {
                                            name: name.clone(),
                                            reason:
                                                FunctionCallSkipReason::PreviousFunctionSuspended,
                                        });
                                    }
                                    Self::push_turn_items(&mut thread, &turn_id, func_items)?;
                                    Self::set_turn_status(
                                        &mut thread,
                                        &turn_id,
                                        TurnStatus::Suspended,
                                    )?;
                                    thread = self.commit_thread(thread, lease.fence()).await?;
                                    self.record_trace(
                                        &mut trace_sequence,
                                        thread_id,
                                        &turn_id,
                                        TraceEventKind::SuspensionCreated {
                                            suspension: suspension.clone(),
                                        },
                                    );
                                    for (skipped_call_id, skipped_name, error) in &skipped_calls {
                                        self.record_trace(
                                            &mut trace_sequence,
                                            thread_id,
                                            &turn_id,
                                            TraceEventKind::ToolOutput {
                                                call_id: skipped_call_id.clone(),
                                                name: skipped_name.clone(),
                                                result: ToolResult::Error {
                                                    error: error.clone(),
                                                },
                                            },
                                        );
                                    }
                                    on_event(TurnStreamEvent::State(TurnStateEvent::Suspended {
                                        suspension: suspension.clone(),
                                    }));
                                    for (skipped_call_id, skipped_name, error) in skipped_calls {
                                        on_event(TurnStreamEvent::State(
                                            TurnStateEvent::FunctionFailed {
                                                call_id: skipped_call_id,
                                                name: skipped_name,
                                                error,
                                            },
                                        ));
                                    }
                                    break 'turn_loop TurnOutcome::Suspended { suspension };
                                }
                                Ok(FunctionCallExecution::SuspendedAfterExecution {
                                    suspension,
                                    output,
                                    effects,
                                }) => {
                                    let update_items =
                                        Self::apply_runtime_effects(&thread, effects);
                                    let hook_result = FunctionCallHookResult::Suspended {
                                        suspension: suspension.clone(),
                                        output: output.clone(),
                                    };
                                    let trace_tool_output = TraceEventKind::ToolOutput {
                                        call_id: call_id.clone(),
                                        name: name.clone(),
                                        result: ToolResult::Success {
                                            output: output.clone(),
                                        },
                                    };
                                    let mut func_items = update_items;
                                    func_items.push(TurnItem::new(
                                        TurnItemSource::Tool,
                                        TurnItemKind::ToolOutput {
                                            call_id: call_id.clone(),
                                            name: name.clone(),
                                            result: ToolResult::Success { output },
                                        },
                                    ));
                                    func_items.push(TurnItem::new(
                                        TurnItemSource::Runtime,
                                        TurnItemKind::SuspensionCreated {
                                            suspension: suspension.clone(),
                                        },
                                    ));
                                    let skipped_calls = calls
                                        .iter()
                                        .skip(call_index + 1)
                                        .map(|skipped_call| {
                                            let error = "function not executed because a previous function suspended the turn";
                                            func_items.push(TurnItem::new(
                                                TurnItemSource::Tool,
                                                TurnItemKind::ToolOutput {
                                                    call_id: skipped_call.call_id.clone(),
                                                    name: skipped_call.name.clone(),
                                                    result: ToolResult::Error {
                                                        error: error.to_string(),
                                                    },
                                                },
                                            ));
                                            (
                                                skipped_call.call_id.clone(),
                                                skipped_call.name.clone(),
                                                error.to_string(),
                                            )
                                        })
                                        .collect::<Vec<_>>();
                                    for (_, name, _) in &skipped_calls {
                                        self.record_metric(RuntimeMetric::FunctionCallSkipped {
                                            name: name.clone(),
                                            reason:
                                                FunctionCallSkipReason::PreviousFunctionSuspended,
                                        });
                                    }
                                    Self::push_turn_items(&mut thread, &turn_id, func_items)?;
                                    Self::set_turn_status(
                                        &mut thread,
                                        &turn_id,
                                        TurnStatus::Suspended,
                                    )?;
                                    thread = self.commit_thread(thread, lease.fence()).await?;
                                    self.record_trace(
                                        &mut trace_sequence,
                                        thread_id,
                                        &turn_id,
                                        TraceEventKind::SuspensionCreated {
                                            suspension: suspension.clone(),
                                        },
                                    );
                                    self.record_trace(
                                        &mut trace_sequence,
                                        thread_id,
                                        &turn_id,
                                        trace_tool_output,
                                    );
                                    for (skipped_call_id, skipped_name, error) in &skipped_calls {
                                        self.record_trace(
                                            &mut trace_sequence,
                                            thread_id,
                                            &turn_id,
                                            TraceEventKind::ToolOutput {
                                                call_id: skipped_call_id.clone(),
                                                name: skipped_name.clone(),
                                                result: ToolResult::Error {
                                                    error: error.clone(),
                                                },
                                            },
                                        );
                                    }
                                    hook_context.projection =
                                        ThreadProjection::from_thread(&thread);
                                    self.run_after_function_call_hooks(
                                        &entered_hooks,
                                        &hook_context,
                                        hook_result,
                                        &mut on_event,
                                    )
                                    .await;
                                    on_event(TurnStreamEvent::State(
                                        TurnStateEvent::FunctionCompleted {
                                            call_id: call_id.clone(),
                                            name: name.clone(),
                                        },
                                    ));
                                    on_event(TurnStreamEvent::State(TurnStateEvent::Suspended {
                                        suspension: suspension.clone(),
                                    }));
                                    for (skipped_call_id, skipped_name, error) in skipped_calls {
                                        on_event(TurnStreamEvent::State(
                                            TurnStateEvent::FunctionFailed {
                                                call_id: skipped_call_id,
                                                name: skipped_name,
                                                error,
                                            },
                                        ));
                                    }
                                    break 'turn_loop TurnOutcome::Suspended { suspension };
                                }
                                Err(error) => {
                                    let error_text = error.to_string();
                                    let hook_result = FunctionCallHookResult::Failed {
                                        error: error_text.clone(),
                                    };
                                    let trace_tool_output = TraceEventKind::ToolOutput {
                                        call_id: call_id.clone(),
                                        name: name.clone(),
                                        result: ToolResult::Error {
                                            error: error_text.clone(),
                                        },
                                    };
                                    Self::push_turn_items(
                                        &mut thread,
                                        &turn_id,
                                        vec![TurnItem::new(
                                            TurnItemSource::Tool,
                                            TurnItemKind::ToolOutput {
                                                call_id: call_id.clone(),
                                                name: name.clone(),
                                                result: ToolResult::Error {
                                                    error: error_text.clone(),
                                                },
                                            },
                                        )],
                                    )?;
                                    thread = self.commit_thread(thread, lease.fence()).await?;
                                    self.record_trace(
                                        &mut trace_sequence,
                                        thread_id,
                                        &turn_id,
                                        trace_tool_output,
                                    );
                                    hook_context.projection =
                                        ThreadProjection::from_thread(&thread);
                                    self.run_after_function_call_hooks(
                                        &entered_hooks,
                                        &hook_context,
                                        hook_result,
                                        &mut on_event,
                                    )
                                    .await;
                                    on_event(TurnStreamEvent::State(
                                        TurnStateEvent::FunctionFailed {
                                            call_id,
                                            name,
                                            error: error_text,
                                        },
                                    ));
                                }
                            }
                        }

                        if admitted_call_count < calls.len() {
                            let error =
                                AgentError::MaxFunctionCalls(limits.max_function_calls).to_string();
                            let skipped_error =
                                format!("function call not executed because {error}");
                            let skipped_calls = calls
                                .iter()
                                .skip(admitted_call_count)
                                .map(|skipped_call| {
                                    (skipped_call.call_id.clone(), skipped_call.name.clone())
                                })
                                .collect::<Vec<_>>();
                            for (_, name) in &skipped_calls {
                                self.record_metric(RuntimeMetric::FunctionCallSkipped {
                                    name: name.clone(),
                                    reason: FunctionCallSkipReason::MaxCallsPerTurn,
                                });
                            }
                            Self::push_turn_items(
                                &mut thread,
                                &turn_id,
                                skipped_calls
                                    .iter()
                                    .map(|(call_id, name)| {
                                        TurnItem::new(
                                            TurnItemSource::Tool,
                                            TurnItemKind::ToolOutput {
                                                call_id: call_id.clone(),
                                                name: name.clone(),
                                                result: ToolResult::Error {
                                                    error: skipped_error.clone(),
                                                },
                                            },
                                        )
                                    })
                                    .collect(),
                            )?;
                            self.fail_turn(&mut thread, &turn_id, error.clone())?;
                            thread = self.commit_thread(thread, lease.fence()).await?;
                            for (call_id, name) in skipped_calls {
                                self.record_trace(
                                    &mut trace_sequence,
                                    thread_id,
                                    &turn_id,
                                    TraceEventKind::ToolOutput {
                                        call_id: call_id.clone(),
                                        name: name.clone(),
                                        result: ToolResult::Error {
                                            error: skipped_error.clone(),
                                        },
                                    },
                                );
                                on_event(TurnStreamEvent::State(TurnStateEvent::FunctionFailed {
                                    call_id,
                                    name,
                                    error: skipped_error.clone(),
                                }));
                            }
                            tracing::warn!(error, "turn exceeded max function calls");
                            break 'turn_loop TurnOutcome::Failed { error };
                        }
                    }
                    ModelResponse::AssistantMessage { .. }
                    | ModelResponse::FunctionCalls { .. } => {
                        unreachable!("legacy model response variants are normalized above")
                    }
                }
            }

            let error = AgentError::MaxIterations(limits.max_model_iterations).to_string();
            tracing::warn!(error, "turn exceeded max iterations");
            self.fail_turn(&mut thread, &turn_id, error.clone())?;
            break 'turn_loop TurnOutcome::Failed { error };
        };

        Self::apply_turn_token_usage(&mut thread, turn_token_usage);
        self.record_metric(RuntimeMetric::TokenUsage {
            input_tokens: turn_token_usage.input_tokens,
            cached_input_tokens: turn_token_usage.cached_input_tokens,
            output_tokens: turn_token_usage.output_tokens,
            total_tokens: turn_token_usage.total_tokens,
        });
        self.commit_thread(thread, lease.fence()).await?;
        let trace_status = match &outcome {
            TurnOutcome::AssistantMessage { .. } => TraceTurnStatus::Completed,
            TurnOutcome::Suspended { .. } => TraceTurnStatus::Suspended,
            TurnOutcome::Failed { .. } => TraceTurnStatus::Failed,
            TurnOutcome::Aborted { .. } => TraceTurnStatus::Aborted,
        };
        self.record_metric(RuntimeMetric::TurnFinished {
            status: match &outcome {
                TurnOutcome::AssistantMessage { .. } => MetricStatus::Completed,
                TurnOutcome::Suspended { .. } => MetricStatus::Suspended,
                TurnOutcome::Failed { .. } => MetricStatus::Failed,
                TurnOutcome::Aborted { .. } => MetricStatus::Aborted,
            },
            duration: turn_started.elapsed(),
            function_calls: function_call_count,
        });
        self.record_trace(
            &mut trace_sequence,
            thread_id,
            &turn_id,
            TraceEventKind::TurnFinished {
                status: trace_status,
            },
        );
        if let TurnOutcome::Failed { error } = &outcome {
            on_event(TurnStreamEvent::State(TurnStateEvent::TurnFailed {
                error: error.clone(),
            }));
        }
        if let TurnOutcome::Aborted { reason } = &outcome {
            on_event(TurnStreamEvent::State(TurnStateEvent::TurnAborted {
                reason: reason.clone(),
            }));
        }
        Self::emit_turn_token_usage(&mut on_event, turn_token_usage);
        on_event(TurnStreamEvent::State(TurnStateEvent::TurnFinished {
            outcome: outcome.clone(),
        }));
        Ok(outcome)
    }

    fn apply_turn_token_usage(thread: &mut Thread, usage: TokenUsage) {
        if !usage.is_zero() {
            thread.token_usage.add_assign(usage);
        }
    }

    fn record_trace(
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

    async fn recover_pending_function_calls(
        &self,
        thread: &mut Thread,
        turn_id: &str,
        abort_signal: &mut TurnAbortSignal,
        lease: &SessionLease,
        on_event: &mut TurnEventHandler<'_>,
        trace_sequence: &mut u64,
    ) -> Result<Option<TurnOutcome>> {
        let thread_id = thread.id.clone();
        let pending_calls = Self::pending_function_calls(thread, turn_id);
        if pending_calls.is_empty() {
            return Ok(None);
        }

        for (call_index, (call, started_attempt)) in pending_calls.iter().enumerate() {
            let attempt = started_attempt.unwrap_or(0).saturating_add(1);
            let recovery_policy = self.function_registry.recovery_policy(&call.name)?;
            self.record_trace(
                trace_sequence,
                &thread_id,
                turn_id,
                TraceEventKind::RecoveryStarted {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    attempt,
                    policy: recovery_policy,
                },
            );
            if started_attempt.is_some()
                && matches!(recovery_policy, FunctionRecoveryPolicy::NonIdempotent)
            {
                let suspension = Suspension {
                    id: new_id("suspension"),
                    kind: SuspensionKind::FunctionRecovery,
                    payload: serde_json::json!({
                        "call_id": call.call_id,
                        "name": call.name,
                        "attempt": attempt,
                        "deferred": true,
                        "recovery": true,
                        "reason": "function started but no terminal output was recorded"
                    }),
                };
                Self::push_turn_items(
                    thread,
                    turn_id,
                    vec![TurnItem::new(
                        TurnItemSource::Runtime,
                        TurnItemKind::SuspensionCreated {
                            suspension: suspension.clone(),
                        },
                    )],
                )?;
                Self::set_turn_status(thread, turn_id, TurnStatus::Suspended)?;
                *thread = self.commit_thread(thread.clone(), lease.fence()).await?;
                self.record_trace(
                    trace_sequence,
                    &thread_id,
                    turn_id,
                    TraceEventKind::SuspensionCreated {
                        suspension: suspension.clone(),
                    },
                );
                on_event(TurnStreamEvent::State(TurnStateEvent::Suspended {
                    suspension: suspension.clone(),
                }));
                return Ok(Some(TurnOutcome::Suspended { suspension }));
            }

            Self::push_turn_items(
                thread,
                turn_id,
                vec![TurnItem::new(
                    TurnItemSource::Runtime,
                    TurnItemKind::FunctionCallStarted {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        attempt,
                    },
                )],
            )?;
            *thread = self.commit_thread(thread.clone(), lease.fence()).await?;

            on_event(TurnStreamEvent::State(TurnStateEvent::FunctionStarted {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
            }));
            self.record_trace(
                trace_sequence,
                &thread_id,
                turn_id,
                TraceEventKind::FunctionCall {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                },
            );

            let context = FunctionContext {
                thread_id: thread.id.clone(),
                metadata: thread.metadata.clone(),
                turn_id: turn_id.to_string(),
                call_id: call.call_id.clone(),
                projection: ThreadProjection::from_thread(thread),
                abort_signal: abort_signal.clone(),
            };
            let function_started = Instant::now();
            let execution = self
                .await_step_or_abort(
                    self.function_registry
                        .call(&call.name, call.arguments.clone(), context),
                    abort_signal,
                    thread,
                    &thread_id,
                    turn_id,
                    "recovered function call",
                )
                .await?;
            self.record_metric(RuntimeMetric::FunctionCallFinished {
                name: call.name.clone(),
                outcome: match &execution {
                    None => FunctionCallOutcome::Aborted,
                    Some(Ok(FunctionCallExecution::Completed { .. })) => {
                        FunctionCallOutcome::Completed
                    }
                    Some(Ok(FunctionCallExecution::SuspendedBeforeExecution { .. }))
                    | Some(Ok(FunctionCallExecution::SuspendedAfterExecution { .. })) => {
                        FunctionCallOutcome::Suspended
                    }
                    Some(Err(_)) => FunctionCallOutcome::Failed,
                },
                duration: function_started.elapsed(),
            });
            let Some(execution) = execution else {
                for (skipped_call, _) in pending_calls.iter().skip(call_index + 1) {
                    self.record_metric(RuntimeMetric::FunctionCallSkipped {
                        name: skipped_call.name.clone(),
                        reason: FunctionCallSkipReason::TurnAborted,
                    });
                }
                Self::append_aborted_function_outputs(
                    thread,
                    turn_id,
                    &pending_calls
                        .iter()
                        .skip(call_index)
                        .map(|(call, _)| call.clone())
                        .collect::<Vec<_>>(),
                    0,
                )?;
                *thread = self.commit_thread(thread.clone(), lease.fence()).await?;
                return Ok(Some(TurnOutcome::Aborted {
                    reason: "turn aborted by caller".to_string(),
                }));
            };

            match execution {
                Ok(FunctionCallExecution::Completed { output, effects }) => {
                    let mut items = Self::apply_runtime_effects(thread, effects);
                    items.push(TurnItem::new(
                        TurnItemSource::Tool,
                        TurnItemKind::ToolOutput {
                            call_id: call.call_id.clone(),
                            name: call.name.clone(),
                            result: ToolResult::Success {
                                output: output.clone(),
                            },
                        },
                    ));
                    Self::push_turn_items(thread, turn_id, items)?;
                    *thread = self.commit_thread(thread.clone(), lease.fence()).await?;
                    self.record_trace(
                        trace_sequence,
                        &thread_id,
                        turn_id,
                        TraceEventKind::ToolOutput {
                            call_id: call.call_id.clone(),
                            name: call.name.clone(),
                            result: ToolResult::Success { output },
                        },
                    );
                    on_event(TurnStreamEvent::State(TurnStateEvent::FunctionCompleted {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                    }));
                }
                Ok(FunctionCallExecution::SuspendedBeforeExecution { suspension, .. })
                | Ok(FunctionCallExecution::SuspendedAfterExecution { suspension, .. }) => {
                    let mut items = vec![TurnItem::new(
                        TurnItemSource::Runtime,
                        TurnItemKind::SuspensionCreated {
                            suspension: suspension.clone(),
                        },
                    )];
                    for (skipped_call, _) in pending_calls.iter().skip(call_index + 1) {
                        items.push(TurnItem::new(
                            TurnItemSource::Tool,
                            TurnItemKind::ToolOutput {
                                call_id: skipped_call.call_id.clone(),
                                name: skipped_call.name.clone(),
                                result: ToolResult::Error {
                                    error: "function not executed because a previous function suspended the turn".to_string(),
                                },
                            },
                        ));
                    }
                    Self::push_turn_items(thread, turn_id, items)?;
                    Self::set_turn_status(thread, turn_id, TurnStatus::Suspended)?;
                    *thread = self.commit_thread(thread.clone(), lease.fence()).await?;
                    self.record_trace(
                        trace_sequence,
                        &thread_id,
                        turn_id,
                        TraceEventKind::SuspensionCreated {
                            suspension: suspension.clone(),
                        },
                    );
                    on_event(TurnStreamEvent::State(TurnStateEvent::Suspended {
                        suspension: suspension.clone(),
                    }));
                    return Ok(Some(TurnOutcome::Suspended { suspension }));
                }
                Err(error) => {
                    let error_text = error.to_string();
                    Self::push_turn_items(
                        thread,
                        turn_id,
                        vec![TurnItem::new(
                            TurnItemSource::Tool,
                            TurnItemKind::ToolOutput {
                                call_id: call.call_id.clone(),
                                name: call.name.clone(),
                                result: ToolResult::Error {
                                    error: error_text.clone(),
                                },
                            },
                        )],
                    )?;
                    *thread = self.commit_thread(thread.clone(), lease.fence()).await?;
                    on_event(TurnStreamEvent::State(TurnStateEvent::FunctionFailed {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        error: error_text,
                    }));
                }
            }
        }
        Ok(None)
    }

    fn pending_function_calls(
        thread: &Thread,
        turn_id: &str,
    ) -> Vec<(ModelFunctionCall, Option<u32>)> {
        let Some(turn) = thread.turns.iter().find(|turn| turn.id == turn_id) else {
            return Vec::new();
        };
        let Some((response_index, calls)) =
            turn.items
                .iter()
                .enumerate()
                .rev()
                .find_map(|(index, item)| match &item.kind {
                    TurnItemKind::ModelResponse { function_calls, .. }
                        if !function_calls.is_empty() =>
                    {
                        Some((index, function_calls.clone()))
                    }
                    _ => None,
                })
        else {
            return Vec::new();
        };
        let mut started = HashMap::new();
        let mut completed = std::collections::HashSet::new();
        for item in turn.items.iter().skip(response_index + 1) {
            match &item.kind {
                TurnItemKind::FunctionCallStarted {
                    call_id, attempt, ..
                } => {
                    started.insert(call_id.clone(), *attempt);
                }
                TurnItemKind::ToolOutput { call_id, .. } => {
                    completed.insert(call_id.clone());
                }
                _ => {}
            }
        }
        calls
            .into_iter()
            .filter(|call| !completed.contains(&call.call_id))
            .map(|call| {
                let attempt = started.get(&call.call_id).copied();
                (call, attempt)
            })
            .collect()
    }

    async fn await_step_or_abort<T, F>(
        &self,
        future: F,
        abort_signal: &mut TurnAbortSignal,
        thread: &mut Thread,
        thread_id: &str,
        turn_id: &str,
        step: &str,
    ) -> Result<Option<Result<T>>>
    where
        F: Future<Output = Result<T>> + Send,
    {
        if abort_signal.is_cancelled() {
            let reason = "turn aborted by caller".to_string();
            tracing::info!(thread_id, turn_id, step, "turn aborted before step");
            self.abort_turn(thread, turn_id, reason)?;
            return Ok(None);
        }

        tokio::select! {
            biased;
            () = abort_signal.wait_cancelled() => {
                let reason = "turn aborted by caller".to_string();
                tracing::info!(thread_id, turn_id, step, "turn aborted");
                self.abort_turn(thread, turn_id, reason)?;
                Ok(None)
            }
            result = future => Ok(Some(result)),
        }
    }

    fn emit_turn_token_usage(on_event: &mut TurnEventHandler<'_>, usage: TokenUsage) {
        if !usage.is_zero() {
            on_event(TurnStreamEvent::State(TurnStateEvent::TurnTokenUsage {
                usage,
            }));
        }
    }

    async fn run_before_function_call_hooks(
        &self,
        context: &FunctionCallHookContext,
        on_event: &mut TurnEventHandler<'_>,
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

    async fn run_after_function_call_hooks(
        &self,
        hooks: &[Arc<dyn FunctionCallHook>],
        context: &FunctionCallHookContext,
        result: FunctionCallHookResult,
        on_event: &mut TurnEventHandler<'_>,
    ) {
        for hook in hooks.iter().rev() {
            if let Err(error) = hook
                .after_call((*context).clone(), result.clone(), on_event)
                .await
            {
                tracing::warn!(error = %error, "function call post-hook failed");
                on_event(TurnStreamEvent::Runtime(RuntimeEvent {
                    source: "function_call_hook".to_string(),
                    message: "post-hook failed".to_string(),
                    metadata: serde_json::json!({
                        "call_id": context.call_id,
                        "name": context.name,
                        "error": error.to_string(),
                    }),
                }));
            }
        }
    }

    async fn commit_thread(
        &self,
        thread: Thread,
        lease_fence: &crate::session::LeaseFence,
    ) -> Result<Thread> {
        self.store.compare_and_commit(thread, lease_fence).await
    }

    async fn model_request_from_projection(
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

    fn apply_runtime_effects(thread: &Thread, effects: Vec<RuntimeEffect>) -> Vec<TurnItem> {
        effects
            .into_iter()
            .map(|effect| match effect {
                RuntimeEffect::SetGoal(goal) => {
                    let previous = ThreadProjection::from_thread(thread).goal;
                    TurnItem::new(
                        TurnItemSource::Runtime,
                        TurnItemKind::GoalUpdated {
                            previous,
                            current: goal,
                        },
                    )
                }
            })
            .collect()
    }

    fn push_turn_items(thread: &mut Thread, turn_id: &str, items: Vec<TurnItem>) -> Result<()> {
        let turn = thread
            .turn_mut(turn_id)
            .ok_or_else(|| AgentError::TurnNotFound(turn_id.to_string()))?;
        for item in items {
            turn.push_item(item);
        }
        Ok(())
    }

    fn set_turn_status(thread: &mut Thread, turn_id: &str, status: TurnStatus) -> Result<()> {
        let turn = thread
            .turn_mut(turn_id)
            .ok_or_else(|| AgentError::TurnNotFound(turn_id.to_string()))?;
        turn.set_status(status);
        Ok(())
    }

    fn append_aborted_function_outputs(
        thread: &mut Thread,
        turn_id: &str,
        calls: &[ModelFunctionCall],
        first_unfinished: usize,
    ) -> Result<()> {
        let items = calls
            .iter()
            .skip(first_unfinished)
            .map(|call| {
                TurnItem::new(
                    TurnItemSource::Tool,
                    TurnItemKind::ToolOutput {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        result: ToolResult::Aborted {
                            reason: "function execution aborted before a result was recorded"
                                .to_string(),
                        },
                    },
                )
            })
            .collect();
        Self::push_turn_items(thread, turn_id, items)
    }

    fn fail_turn(&self, thread: &mut Thread, turn_id: &str, error: String) -> Result<()> {
        Self::push_turn_items(
            thread,
            turn_id,
            vec![TurnItem::new(
                TurnItemSource::Runtime,
                TurnItemKind::TurnFailed { error },
            )],
        )?;
        Self::set_turn_status(thread, turn_id, TurnStatus::Failed)
    }

    fn abort_turn(&self, thread: &mut Thread, turn_id: &str, reason: String) -> Result<()> {
        Self::push_turn_items(
            thread,
            turn_id,
            vec![TurnItem::new(
                TurnItemSource::Runtime,
                TurnItemKind::TurnAborted { reason },
            )],
        )?;
        Self::set_turn_status(thread, turn_id, TurnStatus::Aborted)
    }
}

#[cfg(test)]
mod tests {
    use crate::functions::{builtin_registry, FunctionRegistry, SimpleFunction};
    use crate::model::{
        ModelClient, ModelFunctionCall, ModelRequest, ModelResponse, ModelStreamEvent,
        ModelStreamHandler,
    };
    use crate::store::{ThreadContextCache, ThreadStore};
    use crate::trace::{TraceCollector, TraceEvent, TraceEventKind};
    use crate::{
        Agent, AgentConfig, AgentError, FunctionCallHook, FunctionCallHookContext,
        FunctionCallHookResult, FunctionExecution, FunctionLimits, FunctionRecoveryPolicy,
        FunctionSpec, Result, RuntimeEvent, TurnExecutionLimits, TurnOutcome, TurnStateEvent,
        TurnStreamEvent,
    };
    use canary_agent_kernel::events::{
        Suspension, SuspensionKind, Thread, TokenUsage, ToolResult, Turn, TurnItem, TurnItemKind,
        TurnItemSource, TurnStatus,
    };
    use canary_agent_kernel::projection::ThreadProjection;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    #[derive(Default)]
    struct RecordingTraceCollector {
        events: Mutex<Vec<TraceEvent>>,
    }

    impl TraceCollector for RecordingTraceCollector {
        fn record(&self, event: TraceEvent) {
            self.events.lock().expect("trace events").push(event);
        }
    }

    #[derive(Default)]
    struct TestStore {
        threads: Mutex<std::collections::BTreeMap<String, canary_agent_kernel::events::Thread>>,
        caches: Mutex<std::collections::BTreeMap<String, ThreadContextCache>>,
    }

    impl ThreadStore for TestStore {
        fn load<'a>(
            &'a self,
            thread_id: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<canary_agent_kernel::events::Thread>> + Send + 'a>>
        {
            Box::pin(async move {
                self.threads
                    .lock()
                    .expect("threads")
                    .get(thread_id)
                    .cloned()
                    .ok_or_else(|| AgentError::ThreadNotFound(thread_id.to_string()))
            })
        }

        fn delete<'a>(
            &'a self,
            thread_id: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async move {
                let removed = self.threads.lock().expect("threads").remove(thread_id);
                self.caches.lock().expect("caches").remove(thread_id);
                removed
                    .map(|_| ())
                    .ok_or_else(|| AgentError::ThreadNotFound(thread_id.to_string()))
            })
        }

        fn compare_and_commit<'a>(
            &'a self,
            mut thread: canary_agent_kernel::events::Thread,
            _lease_fence: &'a crate::session::LeaseFence,
        ) -> Pin<Box<dyn Future<Output = Result<canary_agent_kernel::events::Thread>> + Send + 'a>>
        {
            Box::pin(async move {
                let mut threads = self.threads.lock().expect("threads");
                let current_revision = threads
                    .get(&thread.id)
                    .map_or_else(canary_agent_kernel::RevisionToken::initial, |current| {
                        current.revision.clone()
                    });
                if current_revision != thread.revision {
                    return Err(AgentError::RevisionConflict {
                        thread_id: thread.id.clone(),
                        expected: thread.revision.clone(),
                        actual: current_revision,
                    });
                }
                let next = thread
                    .revision
                    .as_bytes()
                    .try_into()
                    .map(u64::from_be_bytes)
                    .unwrap_or(0)
                    .saturating_add(1);
                thread.revision = canary_agent_kernel::RevisionToken::from_u64(next);
                thread.touch();
                threads.insert(thread.id.clone(), thread.clone());
                Ok(thread)
            })
        }

        fn load_context_cache<'a>(
            &'a self,
            thread_id: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<Option<ThreadContextCache>>> + Send + 'a>> {
            Box::pin(async move { Ok(self.caches.lock().expect("caches").get(thread_id).cloned()) })
        }

        fn save_context_cache<'a>(
            &'a self,
            cache: ThreadContextCache,
        ) -> Pin<Box<dyn Future<Output = Result<ThreadContextCache>> + Send + 'a>> {
            Box::pin(async move {
                self.caches
                    .lock()
                    .expect("caches")
                    .insert(cache.thread_id.clone(), cache.clone());
                Ok(cache)
            })
        }
    }

    impl TestStore {
        fn insert_thread(&self, thread: Thread) {
            self.threads
                .lock()
                .expect("threads")
                .insert(thread.id.clone(), thread);
        }
    }

    struct MockModel {
        responses: Mutex<VecDeque<ModelResponse>>,
        requests: Mutex<Vec<ModelRequest>>,
    }

    impl MockModel {
        fn new(responses: Vec<ModelResponse>) -> Self {
            Self {
                responses: Mutex::new(VecDeque::from(responses)),
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    impl ModelClient for MockModel {
        fn stream_complete<'a>(
            &'a self,
            request: ModelRequest,
            _on_event: &'a mut ModelStreamHandler<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
            Box::pin(async move {
                self.requests.lock().expect("requests").push(request);
                self.responses
                    .lock()
                    .expect("lock")
                    .pop_front()
                    .ok_or_else(|| crate::AgentError::Model("no mock response".to_string()))
            })
        }
    }

    struct PendingModel;

    impl ModelClient for PendingModel {
        fn stream_complete<'a>(
            &'a self,
            _request: ModelRequest,
            _on_event: &'a mut ModelStreamHandler<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
            Box::pin(async move { std::future::pending::<Result<ModelResponse>>().await })
        }
    }

    fn agent_with(store: Arc<dyn ThreadStore>, responses: Vec<ModelResponse>) -> Agent {
        Agent::new(
            AgentConfig::default(),
            store,
            Arc::new(MockModel::new(responses)),
            test_registry(),
            Arc::new(crate::session::LocalSessionCoordinator::default()),
        )
    }

    fn test_registry() -> FunctionRegistry {
        let mut registry = builtin_registry();
        registry.register(SimpleFunction::new(
            FunctionSpec {
                name: "test_function".to_string(),
                description: "Test-only function.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
            },
            FunctionLimits {
                time_budget: Duration::from_secs(1),
                max_output_bytes: 20 * 1024 * 1024,
            },
            FunctionRecoveryPolicy::Idempotent,
            |_args, _context| async {
                Ok(FunctionExecution::Completed {
                    output: json!({ "ok": true }),
                })
            },
        ));
        registry.register(SimpleFunction::new(
            FunctionSpec {
                name: "test_suspend".to_string(),
                description: "Test-only suspension function.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
            },
            FunctionLimits {
                time_budget: Duration::from_secs(1),
                max_output_bytes: 20 * 1024 * 1024,
            },
            FunctionRecoveryPolicy::NonIdempotent,
            |_args, _context| async {
                Ok(FunctionExecution::SuspendedAfterExecution {
                    suspension: Suspension {
                        id: "test-suspension".to_string(),
                        kind: SuspensionKind::LongRunningJob,
                        payload: json!({}),
                    },
                    output: json!({ "status": "suspended" }),
                })
            },
        ));
        registry
    }

    struct RecordingHook {
        label: &'static str,
        events: Arc<Mutex<Vec<String>>>,
        fail_before: bool,
        fail_after: bool,
    }

    impl RecordingHook {
        fn new(label: &'static str, events: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                label,
                events,
                fail_before: false,
                fail_after: false,
            }
        }

        fn fail_before(mut self) -> Self {
            self.fail_before = true;
            self
        }

        fn fail_after(mut self) -> Self {
            self.fail_after = true;
            self
        }
    }

    impl FunctionCallHook for RecordingHook {
        fn before_call<'a>(
            &'a self,
            context: FunctionCallHookContext,
            emit: &'a mut super::TurnEventHandler<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async move {
                self.events
                    .lock()
                    .expect("events")
                    .push(format!("before:{}:{}", self.label, context.name));
                emit(TurnStreamEvent::Runtime(RuntimeEvent {
                    source: self.label.to_string(),
                    message: "before".to_string(),
                    metadata: json!({ "name": context.name }),
                }));
                if self.fail_before {
                    return Err(AgentError::Function {
                        name: context.name,
                        message: format!("{} blocked call", self.label),
                    });
                }
                Ok(())
            })
        }

        fn after_call<'a>(
            &'a self,
            context: FunctionCallHookContext,
            result: FunctionCallHookResult,
            _emit: &'a mut super::TurnEventHandler<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async move {
                let result_label = match result {
                    FunctionCallHookResult::Completed { .. } => "completed",
                    FunctionCallHookResult::Suspended { .. } => "waiting",
                    FunctionCallHookResult::Failed { .. } => "failed",
                };
                self.events.lock().expect("events").push(format!(
                    "after:{}:{}:{result_label}",
                    self.label, context.name
                ));
                if self.fail_after {
                    return Err(AgentError::Function {
                        name: context.name,
                        message: format!("{} post-hook failed", self.label),
                    });
                }
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn simple_assistant_message_ends_turn() {
        let store = Arc::new(TestStore::default());
        let agent = agent_with(
            store.clone(),
            vec![ModelResponse::AssistantMessage {
                text: "hello".to_string(),
            }],
        );

        let outcome = agent
            .run_turn("t", "hi", json!({}), |_| {})
            .await
            .expect("turn");
        assert_eq!(
            outcome,
            TurnOutcome::AssistantMessage {
                text: "hello".to_string()
            }
        );
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns.len(), 1);
        assert_eq!(thread.turns[0].status, TurnStatus::Completed);
        assert_eq!(thread.turns[0].items.len(), 2);
    }

    #[tokio::test]
    async fn injects_metadata_when_creating_a_thread() {
        let store = Arc::new(TestStore::default());
        let agent = agent_with(
            store.clone(),
            vec![ModelResponse::AssistantMessage {
                text: "hello".to_string(),
            }],
        );

        agent
            .run_turn("t", "hi", json!({"workspace": "/tmp/project"}), |_| {})
            .await
            .expect("turn");

        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.metadata, json!({"workspace": "/tmp/project"}));
    }

    #[tokio::test]
    async fn empty_model_response_fails_turn() {
        let store = Arc::new(TestStore::default());
        let agent = agent_with(
            store.clone(),
            vec![ModelResponse::Assistant {
                text: None,
                function_calls: Vec::new(),
            }],
        );

        let outcome = agent
            .run_turn("t", "hello", json!({}), |_| {})
            .await
            .expect("turn");
        assert!(
            matches!(outcome, TurnOutcome::Failed { error } if error.contains("neither assistant text nor function calls"))
        );
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns[0].status, TurnStatus::Failed);
    }

    #[tokio::test]
    async fn persists_assistant_text_when_response_also_requests_tools() {
        let store = Arc::new(TestStore::default());
        let agent = agent_with(
            store.clone(),
            vec![
                ModelResponse::Assistant {
                    text: Some("I will check the goal first.".to_string()),
                    function_calls: vec![ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "test_function".to_string(),
                        arguments: json!({}),
                    }],
                },
                ModelResponse::Assistant {
                    text: Some("The goal is not set.".to_string()),
                    function_calls: Vec::new(),
                },
            ],
        );

        agent
            .run_turn("t", "check", json!({}), |_| {})
            .await
            .expect("turn");
        let thread = store.load("t").await.expect("thread");
        let messages = thread.turns[0]
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                TurnItemKind::ModelResponse { text, .. } => text.as_deref(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            messages,
            vec!["I will check the goal first.", "The goal is not set."]
        );
    }

    #[tokio::test]
    async fn abort_during_model_request_persists_aborted_turn() {
        let store = Arc::new(TestStore::default());
        let agent = Arc::new(Agent::new(
            AgentConfig::default(),
            store.clone(),
            Arc::new(PendingModel),
            test_registry(),
            Arc::new(crate::session::LocalSessionCoordinator::default()),
        ));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let started_tx = Arc::new(Mutex::new(Some(started_tx)));
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = events.clone();
        let captured_started = started_tx.clone();
        let turn_agent = agent.clone();

        let turn = tokio::spawn(async move {
            turn_agent
                .run_turn("t", "slow", json!({}), move |event| {
                    if matches!(
                        event,
                        TurnStreamEvent::State(TurnStateEvent::TurnStarted { .. })
                    ) {
                        if let Some(sender) = captured_started.lock().expect("started").take() {
                            let _ = sender.send(());
                        }
                    }
                    captured_events.lock().expect("events").push(event);
                })
                .await
        });

        started_rx.await.expect("turn started");
        agent.abort("t").expect("active turn");
        let outcome = turn.await.expect("join").expect("turn");

        assert!(matches!(outcome, TurnOutcome::Aborted { .. }));
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns.len(), 1);
        assert_eq!(thread.turns[0].status, TurnStatus::Aborted);
        assert!(thread.turns[0]
            .items
            .iter()
            .any(|item| matches!(item.kind, TurnItemKind::TurnAborted { .. })));
        assert!(events.lock().expect("events").iter().any(|event| matches!(
            event,
            TurnStreamEvent::State(TurnStateEvent::TurnAborted { .. })
        )));
    }

    #[tokio::test]
    async fn stream_emits_intermediate_and_final_events() {
        let store = Arc::new(TestStore::default());
        let agent = agent_with(
            store,
            vec![
                ModelResponse::FunctionCalls {
                    calls: vec![ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "test_function".to_string(),
                        arguments: json!({}),
                    }],
                },
                ModelResponse::AssistantMessage {
                    text: "done".to_string(),
                },
            ],
        );
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();

        let outcome = agent
            .run_turn("t", "goal?", json!({}), move |event| {
                captured.lock().expect("lock").push(event);
            })
            .await
            .expect("turn");

        assert_eq!(
            outcome,
            TurnOutcome::AssistantMessage {
                text: "done".to_string()
            }
        );
        let events = events.lock().expect("lock");
        assert!(events.iter().any(|event| matches!(
            event,
            TurnStreamEvent::State(TurnStateEvent::FunctionStarted { .. })
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            TurnStreamEvent::State(TurnStateEvent::TurnFinished { .. })
        )));
    }

    #[tokio::test]
    async fn turn_token_usage_is_emitted_and_persisted_on_thread() {
        struct UsageModel;

        impl ModelClient for UsageModel {
            fn stream_complete<'a>(
                &'a self,
                _request: ModelRequest,
                on_event: &'a mut ModelStreamHandler<'a>,
            ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + 'a>> {
                Box::pin(async move {
                    on_event(ModelStreamEvent::TokenUsage {
                        usage: TokenUsage {
                            input_tokens: 10,
                            cached_input_tokens: 2,
                            output_tokens: 4,
                            total_tokens: 14,
                        },
                    });
                    Ok(ModelResponse::AssistantMessage {
                        text: "done".to_string(),
                    })
                })
            }
        }

        let store = Arc::new(TestStore::default());
        let agent = Agent::new(
            AgentConfig::default(),
            store.clone(),
            Arc::new(UsageModel),
            test_registry(),
            Arc::new(crate::session::LocalSessionCoordinator::default()),
        );
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();

        let outcome = agent
            .run_turn("t", "usage?", json!({}), move |event| {
                captured.lock().expect("lock").push(event);
            })
            .await
            .expect("turn");

        assert!(matches!(outcome, TurnOutcome::AssistantMessage { .. }));
        let usage = {
            let events = events.lock().expect("lock");
            events
                .iter()
                .find_map(|event| match event {
                    TurnStreamEvent::State(TurnStateEvent::TurnTokenUsage { usage }) => {
                        Some(*usage)
                    }
                    _ => None,
                })
                .expect("usage event")
        };
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.cached_input_tokens, 2);
        assert_eq!(usage.output_tokens, 4);
        assert_eq!(usage.total_tokens, 14);

        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.token_usage, usage);
    }

    #[tokio::test]
    async fn update_goal_then_message() {
        let store = Arc::new(TestStore::default());
        let agent = agent_with(
            store.clone(),
            vec![
                ModelResponse::FunctionCalls {
                    calls: vec![ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "update_goal".to_string(),
                        arguments: json!({ "objective": "ship", "status": "active" }),
                    }],
                },
                ModelResponse::AssistantMessage {
                    text: "goal set".to_string(),
                },
            ],
        );

        let outcome = agent
            .run_turn("t", "set a goal", json!({}), |_| {})
            .await
            .expect("turn");
        assert_eq!(
            outcome,
            TurnOutcome::AssistantMessage {
                text: "goal set".to_string()
            }
        );
        let thread = store.load("t").await.expect("thread");
        let projection = ThreadProjection::from_thread(&thread);
        assert_eq!(
            projection.goal.as_ref().map(|goal| goal.objective.as_str()),
            Some("ship")
        );
        assert!(thread.turns[0]
            .items
            .iter()
            .any(|item| matches!(item.kind, TurnItemKind::GoalUpdated { .. })));
    }

    #[tokio::test]
    async fn suspended_after_execution_stops_turn() {
        let store = Arc::new(TestStore::default());
        let agent = agent_with(
            store.clone(),
            vec![ModelResponse::FunctionCalls {
                calls: vec![
                    ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "test_suspend".to_string(),
                        arguments: json!({ "prompt": "Which one?" }),
                    },
                    ModelFunctionCall {
                        call_id: "c2".to_string(),
                        name: "test_function".to_string(),
                        arguments: json!({}),
                    },
                ],
            }],
        );

        let outcome = agent
            .run_turn("t", "compare", json!({}), |_| {})
            .await
            .expect("turn");
        assert!(matches!(outcome, TurnOutcome::Suspended { .. }));
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns[0].status, TurnStatus::Suspended);
        assert!(thread.turns[0]
            .items
            .iter()
            .any(|item| matches!(item.kind, TurnItemKind::SuspensionCreated { .. })));
        assert_eq!(
            thread.turns[0]
                .items
                .iter()
                .filter(|item| matches!(item.kind, TurnItemKind::ToolOutput { .. }))
                .count(),
            2
        );
        let error = agent
            .run_turn("t", "another prompt", json!({}), |_| {})
            .await
            .unwrap_err();
        assert!(matches!(error, AgentError::SuspendedTurn { .. }));
    }

    #[tokio::test]
    async fn function_failure_becomes_tool_error() {
        let store = Arc::new(TestStore::default());
        let agent = agent_with(
            store.clone(),
            vec![
                ModelResponse::FunctionCalls {
                    calls: vec![ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "missing".to_string(),
                        arguments: json!({}),
                    }],
                },
                ModelResponse::AssistantMessage {
                    text: "could not call it".to_string(),
                },
            ],
        );

        let outcome = agent
            .run_turn("t", "call missing", json!({}), |_| {})
            .await
            .expect("turn");
        assert!(matches!(outcome, TurnOutcome::AssistantMessage { .. }));
        let thread = store.load("t").await.expect("thread");
        assert!(thread.turns[0].items.iter().any(|item| matches!(
            &item.kind,
            TurnItemKind::ToolOutput {
                result: ToolResult::Error { .. },
                ..
            }
        )));
    }

    #[tokio::test]
    async fn function_call_hooks_are_stacked_in_order() {
        let store = Arc::new(TestStore::default());
        let hook_events = Arc::new(Mutex::new(Vec::new()));
        let agent = agent_with(
            store,
            vec![
                ModelResponse::FunctionCalls {
                    calls: vec![ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "test_function".to_string(),
                        arguments: json!({}),
                    }],
                },
                ModelResponse::AssistantMessage {
                    text: "done".to_string(),
                },
            ],
        )
        .with_function_call_hook(RecordingHook::new("a", hook_events.clone()))
        .with_function_call_hook(RecordingHook::new("b", hook_events.clone()));

        let outcome = agent
            .run_turn("t", "goal?", json!({}), |_| {})
            .await
            .expect("turn");
        assert!(matches!(outcome, TurnOutcome::AssistantMessage { .. }));

        assert_eq!(
            *hook_events.lock().expect("events"),
            vec![
                "before:a:test_function",
                "before:b:test_function",
                "after:b:test_function:completed",
                "after:a:test_function:completed",
            ]
        );
    }

    #[tokio::test]
    async fn pre_hook_failure_blocks_function_and_records_tool_error() {
        let store = Arc::new(TestStore::default());
        let hook_events = Arc::new(Mutex::new(Vec::new()));
        let agent = agent_with(
            store.clone(),
            vec![
                ModelResponse::FunctionCalls {
                    calls: vec![ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "test_function".to_string(),
                        arguments: json!({}),
                    }],
                },
                ModelResponse::AssistantMessage {
                    text: "blocked".to_string(),
                },
            ],
        )
        .with_function_call_hook(RecordingHook::new("audit", hook_events.clone()))
        .with_function_call_hook(RecordingHook::new("policy", hook_events.clone()).fail_before());

        let outcome = agent
            .run_turn("t", "goal?", json!({}), |_| {})
            .await
            .expect("turn");
        assert!(matches!(outcome, TurnOutcome::AssistantMessage { .. }));
        assert_eq!(
            *hook_events.lock().expect("events"),
            vec![
                "before:audit:test_function",
                "before:policy:test_function",
                "after:audit:test_function:failed",
            ]
        );

        let thread = store.load("t").await.expect("thread");
        let tool_outputs = thread.turns[0]
            .items
            .iter()
            .filter(|item| matches!(item.kind, TurnItemKind::ToolOutput { .. }))
            .count();
        assert_eq!(tool_outputs, 1);
        assert!(thread.turns[0].items.iter().any(|item| matches!(
            &item.kind,
            TurnItemKind::ToolOutput {
                call_id,
                name,
                result: ToolResult::Error { error },
            } if call_id == "c1"
                && name == "test_function"
                && error.contains("policy blocked call")
        )));
    }

    #[tokio::test]
    async fn post_hook_failure_is_non_blocking_runtime_event() {
        let store = Arc::new(TestStore::default());
        let hook_events = Arc::new(Mutex::new(Vec::new()));
        let agent = agent_with(
            store.clone(),
            vec![
                ModelResponse::FunctionCalls {
                    calls: vec![ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "test_function".to_string(),
                        arguments: json!({}),
                    }],
                },
                ModelResponse::AssistantMessage {
                    text: "done".to_string(),
                },
            ],
        )
        .with_function_call_hook(RecordingHook::new("audit", hook_events.clone()).fail_after());
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();

        let outcome = agent
            .run_turn("t", "goal?", json!({}), move |event| {
                captured.lock().expect("events").push(event);
            })
            .await
            .expect("turn");

        assert_eq!(
            outcome,
            TurnOutcome::AssistantMessage {
                text: "done".to_string()
            }
        );
        assert_eq!(
            *hook_events.lock().expect("events"),
            vec![
                "before:audit:test_function",
                "after:audit:test_function:completed"
            ]
        );
        assert!(events.lock().expect("events").iter().any(|event| matches!(
            event,
            TurnStreamEvent::Runtime(RuntimeEvent { source, message, metadata })
                if source == "function_call_hook"
                    && message == "post-hook failed"
                    && metadata["name"] == "test_function"
        )));

        let thread = store.load("t").await.expect("thread");
        assert!(thread.turns[0].items.iter().any(|item| matches!(
            &item.kind,
            TurnItemKind::ToolOutput {
                result: ToolResult::Success { .. },
                ..
            }
        )));
    }

    #[tokio::test]
    async fn max_iterations_fails_turn() {
        let store = Arc::new(TestStore::default());
        let agent = Agent::new(
            AgentConfig {
                turn_execution_limits: TurnExecutionLimits {
                    max_model_iterations: 1,
                    ..TurnExecutionLimits::default()
                },
                ..AgentConfig::default()
            },
            store.clone(),
            Arc::new(MockModel::new(vec![ModelResponse::FunctionCalls {
                calls: vec![ModelFunctionCall {
                    call_id: "c1".to_string(),
                    name: "test_function".to_string(),
                    arguments: json!({}),
                }],
            }])),
            test_registry(),
            Arc::new(crate::session::LocalSessionCoordinator::default()),
        );

        let outcome = agent
            .run_turn("t", "goal?", json!({}), |_| {})
            .await
            .expect("turn");
        assert!(matches!(outcome, TurnOutcome::Failed { .. }));
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns[0].status, TurnStatus::Failed);
        assert!(thread.turns[0]
            .items
            .iter()
            .any(|item| matches!(item.kind, TurnItemKind::TurnFailed { .. })));
    }

    #[tokio::test]
    async fn function_call_limit_records_skipped_calls_and_fails_turn() {
        let store = Arc::new(TestStore::default());
        let agent = Agent::new(
            AgentConfig {
                turn_execution_limits: TurnExecutionLimits {
                    max_model_iterations: 4,
                    max_function_calls: 1,
                },
                ..AgentConfig::default()
            },
            store.clone(),
            Arc::new(MockModel::new(vec![ModelResponse::FunctionCalls {
                calls: vec![
                    ModelFunctionCall {
                        call_id: "c1".to_string(),
                        name: "test_function".to_string(),
                        arguments: json!({}),
                    },
                    ModelFunctionCall {
                        call_id: "c2".to_string(),
                        name: "test_function".to_string(),
                        arguments: json!({}),
                    },
                ],
            }])),
            test_registry(),
            Arc::new(crate::session::LocalSessionCoordinator::default()),
        );

        let outcome = agent
            .run_turn("t", "call twice", json!({}), |_| {})
            .await
            .expect("turn");
        assert!(
            matches!(outcome, TurnOutcome::Failed { ref error } if error.contains("max function calls"))
        );

        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns[0].status, TurnStatus::Failed);
        assert!(thread.turns[0].items.iter().any(|item| {
            matches!(
                &item.kind,
                TurnItemKind::ToolOutput {
                    call_id,
                    result: ToolResult::Success { .. },
                    ..
                } if call_id == "c1"
            )
        }));
        assert!(thread.turns[0].items.iter().any(|item| {
            matches!(
                &item.kind,
                TurnItemKind::ToolOutput {
                    call_id,
                    result: ToolResult::Error { error },
                    ..
                } if call_id == "c2" && error.contains("max function calls")
            )
        }));
    }

    #[tokio::test]
    async fn cancellation_wins_before_an_immediate_function_starts() {
        let store = Arc::new(TestStore::default());
        let agent = agent_with(
            store.clone(),
            vec![ModelResponse::FunctionCalls {
                calls: vec![ModelFunctionCall {
                    call_id: "c1".to_string(),
                    name: "test_function".to_string(),
                    arguments: json!({}),
                }],
            }],
        );
        let aborting_agent = agent.clone();

        let outcome = agent
            .run_turn("t", "run it", json!({}), move |event| {
                if matches!(
                    event,
                    TurnStreamEvent::State(TurnStateEvent::FunctionCallsRequested { .. })
                ) {
                    aborting_agent.abort("t").expect("active turn");
                }
            })
            .await
            .expect("turn");

        assert!(matches!(outcome, TurnOutcome::Aborted { .. }));
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns[0].status, TurnStatus::Aborted);
        assert!(!thread.turns[0].items.iter().any(|item| {
            matches!(
                &item.kind,
                TurnItemKind::ToolOutput {
                    result: ToolResult::Success { .. },
                    ..
                }
            )
        }));
    }

    #[tokio::test]
    async fn recovery_retries_idempotent_started_call_before_next_model_step() {
        let store = Arc::new(TestStore::default());
        let mut thread = Thread::new("t");
        let mut turn = Turn::new();
        let turn_id = turn.id.clone();
        turn.push_item(TurnItem::new(
            TurnItemSource::User,
            TurnItemKind::UserInput {
                text: "continue".to_string(),
                response_to: None,
            },
        ));
        turn.push_item(TurnItem::new(
            TurnItemSource::Model,
            TurnItemKind::ModelResponse {
                text: None,
                function_calls: vec![ModelFunctionCall {
                    call_id: "c1".to_string(),
                    name: "test_function".to_string(),
                    arguments: json!({}),
                }],
            },
        ));
        turn.push_item(TurnItem::new(
            TurnItemSource::Runtime,
            TurnItemKind::FunctionCallStarted {
                call_id: "c1".to_string(),
                name: "test_function".to_string(),
                attempt: 1,
            },
        ));
        assert_eq!(turn.id, turn_id);
        thread.turns.push(turn);
        store.insert_thread(thread);

        let agent = agent_with(
            store.clone(),
            vec![ModelResponse::AssistantMessage {
                text: "recovered".to_string(),
            }],
        );
        let outcome = agent.recover_turn("t", |_| {}).await.expect("recovery");
        assert_eq!(
            outcome,
            TurnOutcome::AssistantMessage {
                text: "recovered".to_string()
            }
        );
        let thread = store.load("t").await.expect("thread");
        assert!(thread.turns[0].items.iter().any(|item| {
            matches!(
                &item.kind,
                TurnItemKind::ToolOutput {
                    call_id,
                    result: ToolResult::Success { .. },
                    ..
                } if call_id == "c1"
            )
        }));
        assert!(thread.turns[0].items.iter().any(|item| {
            matches!(
                &item.kind,
                TurnItemKind::FunctionCallStarted { call_id, attempt, .. }
                    if call_id == "c1" && *attempt == 2
            )
        }));
    }

    #[tokio::test]
    async fn recovery_suspends_non_idempotent_started_call() {
        let store = Arc::new(TestStore::default());
        let mut thread = Thread::new("t");
        let mut turn = Turn::new();
        turn.push_item(TurnItem::new(
            TurnItemSource::Model,
            TurnItemKind::ModelResponse {
                text: None,
                function_calls: vec![ModelFunctionCall {
                    call_id: "c1".to_string(),
                    name: "test_suspend".to_string(),
                    arguments: json!({}),
                }],
            },
        ));
        turn.push_item(TurnItem::new(
            TurnItemSource::Runtime,
            TurnItemKind::FunctionCallStarted {
                call_id: "c1".to_string(),
                name: "test_suspend".to_string(),
                attempt: 1,
            },
        ));
        thread.turns.push(turn);
        store.insert_thread(thread);

        let collector = Arc::new(RecordingTraceCollector::default());
        let agent =
            agent_with(store.clone(), vec![]).with_shared_trace_collector(collector.clone());
        let outcome = agent.recover_turn("t", |_| {}).await.expect("recovery");
        assert!(matches!(outcome, TurnOutcome::Suspended { ref suspension }
            if suspension.kind == SuspensionKind::FunctionRecovery));
        let thread = store.load("t").await.expect("thread");
        assert_eq!(thread.turns[0].status, TurnStatus::Suspended);
        let events = collector.events.lock().expect("trace events");
        assert!(events.iter().any(|event| {
            matches!(
                event.kind,
                TraceEventKind::RecoveryStarted {
                    ref call_id,
                    attempt: 2,
                    ..
                } if call_id == "c1"
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(event.kind, TraceEventKind::SuspensionCreated { ref suspension }
                if suspension.kind == SuspensionKind::FunctionRecovery)
        }));
    }

    #[tokio::test]
    async fn recovery_executes_call_without_a_started_marker() {
        let store = Arc::new(TestStore::default());
        let mut thread = Thread::new("t");
        let mut turn = Turn::new();
        turn.push_item(TurnItem::new(
            TurnItemSource::Model,
            TurnItemKind::ModelResponse {
                text: None,
                function_calls: vec![ModelFunctionCall {
                    call_id: "c1".to_string(),
                    name: "test_function".to_string(),
                    arguments: json!({}),
                }],
            },
        ));
        thread.turns.push(turn);
        store.insert_thread(thread);

        let agent = agent_with(
            store.clone(),
            vec![ModelResponse::AssistantMessage {
                text: "continued".to_string(),
            }],
        );
        let outcome = agent.recover_turn("t", |_| {}).await.expect("recovery");
        assert_eq!(
            outcome,
            TurnOutcome::AssistantMessage {
                text: "continued".to_string()
            }
        );
        let thread = store.load("t").await.expect("thread");
        assert!(thread.turns[0].items.iter().any(|item| {
            matches!(
                &item.kind,
                TurnItemKind::FunctionCallStarted { call_id, attempt, .. }
                    if call_id == "c1" && *attempt == 1
            )
        }));
        assert!(thread.turns[0].items.iter().any(|item| {
            matches!(
                &item.kind,
                TurnItemKind::ToolOutput {
                    call_id,
                    result: ToolResult::Success { .. },
                    ..
                } if call_id == "c1"
            )
        }));
    }
}
