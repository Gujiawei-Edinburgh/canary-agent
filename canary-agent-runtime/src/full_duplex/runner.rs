use super::worker::{self, OperationResult, WorkerEvent};
use super::*;
use canary_agent_kernel::projection::OperationStatus;
use std::collections::HashSet;
use tokio::task::{AbortHandle, Id, JoinSet};

struct Inference {
    id: String,
    generation: u64,
    task: AbortHandle,
    started: Instant,
}

struct Operation {
    call: ModelFunctionCall,
    abort: Option<TurnAbortHandle>,
    suspension: Option<(Suspension, bool, Option<Value>)>,
    attempt: u32,
}

enum Job {
    Model(String),
    Operation(String),
    Hook,
}

pub(super) struct Runner {
    agent: Agent,
    thread: Thread,
    turn_id: String,
    config: FullDuplexConfig,
    input: mpsc::Receiver<Command>,
    output: broadcast::Sender<FullDuplexEvent>,
    guard: ActiveTurnGuard,
    lease: SessionLease,
    jobs: JoinSet<()>,
    job_ids: HashMap<Id, Job>,
    sender: mpsc::Sender<WorkerEvent>,
    receiver: mpsc::Receiver<WorkerEvent>,
    inference: Option<Inference>,
    operations: BTreeMap<String, Operation>,
    generation: u64,
    dirty: bool,
    model_iterations: usize,
    function_calls: usize,
    trace_sequence: u64,
    usage: TokenUsage,
    stopping: Option<TurnOutcome>,
    abort_observed: bool,
}

impl Runner {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        agent: Agent,
        thread: Thread,
        turn_id: String,
        config: FullDuplexConfig,
        input: mpsc::Receiver<Command>,
        output: broadcast::Sender<FullDuplexEvent>,
        guard: ActiveTurnGuard,
        lease: SessionLease,
    ) -> Self {
        let (sender, receiver) = mpsc::channel(256);
        Self {
            agent,
            thread,
            turn_id,
            config,
            input,
            output,
            guard,
            lease,
            jobs: JoinSet::new(),
            job_ids: HashMap::new(),
            sender,
            receiver,
            inference: None,
            operations: BTreeMap::new(),
            generation: 1,
            dirty: true,
            model_iterations: 0,
            function_calls: 0,
            trace_sequence: 0,
            usage: TokenUsage::default(),
            stopping: None,
            abort_observed: false,
        }
    }

    pub(super) async fn run(mut self, recovering: bool) -> Result<TurnOutcome> {
        let started = Instant::now();
        let result = self.run_inner(recovering).await;
        // On infrastructure failure, propagate cancellation before dropping workers.
        if result.is_err() {
            if let Some(inference) = self.inference.take() {
                inference.task.abort();
            }
            for operation in self.operations.values() {
                if let Some(abort) = &operation.abort {
                    abort.abort();
                }
            }
            let grace = self.config.cancellation_grace + Duration::from_millis(100);
            let cleanup = async {
                while self.operations.values().any(|op| op.abort.is_some()) {
                    if let Some(event) = self.receiver.recv().await {
                        let _ = self.worker_event(event).await;
                    } else {
                        break;
                    }
                }
            };
            let _ = tokio::time::timeout(grace, cleanup).await;
        }
        self.jobs.abort_all();
        while self.jobs.join_next().await.is_some() {}
        let status = match &result {
            Ok(TurnOutcome::Aborted { .. }) => MetricStatus::Aborted,
            Ok(TurnOutcome::Failed { .. }) | Err(_) => MetricStatus::Failed,
            _ => MetricStatus::Completed,
        };
        self.agent.record_metric(RuntimeMetric::TurnFinished {
            status,
            duration: started.elapsed(),
            function_calls: self.function_calls,
        });
        if let Ok(outcome) = &result {
            self.emit(FullDuplexEvent::Finished {
                outcome: outcome.clone(),
            });
        }
        self.agent.trace_collector.flush().await;
        result
    }

    async fn run_inner(&mut self, recovering: bool) -> Result<TurnOutcome> {
        if recovering {
            self.recover_operations().await?;
        } else {
            let items = self.turn().items.clone();
            for item in items {
                self.publish_item(item);
            }
        }
        loop {
            if self.guard.signal.is_cancelled() && !self.abort_observed {
                self.abort_observed = true;
                self.begin_stop(TurnOutcome::Aborted {
                    reason: "turn aborted by caller".into(),
                })
                .await?;
            }
            if self.stopping.is_some()
                && self.jobs.is_empty()
                && !self.operations.values().any(|op| op.abort.is_some())
            {
                return self.finish().await;
            }
            // Drain accepted ingress before scheduling or accepting a model result.
            if self.stopping.is_none() {
                if let Ok(command) = self.input.try_recv() {
                    self.command(command).await?;
                    continue;
                }
                if self.inference.is_none() && self.dirty {
                    self.start_inference().await?;
                }
            }
            let mut abort = self.guard.signal.clone();
            tokio::select! {
                biased;
                () = abort.cancelled(), if !self.abort_observed => {},
                command = self.input.recv(), if self.stopping.is_none() => {
                    match command {
                        Some(command) => self.command(command).await?,
                        None => self.begin_stop(TurnOutcome::Aborted { reason: "input channel closed".into() }).await?,
                    }
                }
                event = self.receiver.recv() => {
                    if let Some(event) = event { self.worker_event(event).await?; }
                }
                joined = self.jobs.join_next_with_id(), if !self.jobs.is_empty() => {
                    match joined {
                        Some(Ok((id, ()))) => { self.job_ids.remove(&id); }
                        Some(Err(error)) => {
                            let job = self.job_ids.remove(&error.id());
                            if !error.is_cancelled() {
                                match job {
                                    Some(Job::Operation(operation_id)) => self.indeterminate(&operation_id, format!("worker failed: {error}")).await?,
                                    Some(Job::Model(id)) if self.inference.as_ref().is_some_and(|current| current.id == id) => {
                                        self.begin_stop(TurnOutcome::Failed { error: format!("model worker failed: {error}") }).await?;
                                    }
                                    _ => tracing::warn!(%error, "full-duplex auxiliary worker failed"),
                                }
                            }
                        }
                        None => {}
                    }
                }
            }
        }
    }

    fn turn(&self) -> &Turn {
        self.thread
            .turns
            .iter()
            .find(|turn| turn.id == self.turn_id)
            .expect("owned turn")
    }

    fn emit(&self, event: FullDuplexEvent) {
        let _ = self.output.send(event);
    }

    fn publish_item(&mut self, item: TurnItem) {
        self.agent.record_trace(
            &mut self.trace_sequence,
            &self.thread.id,
            &self.turn_id,
            TraceEventKind::InteractionItem { item: item.clone() },
        );
        self.emit(FullDuplexEvent::ItemCommitted { item });
    }

    async fn append(&mut self, items: Vec<TurnItem>) -> Result<()> {
        let mut thread = self.thread.clone();
        Agent::push_turn_items(&mut thread, &self.turn_id, items.clone())?;
        self.thread = self.agent.commit_thread(thread, self.lease.fence()).await?;
        for item in items {
            self.publish_item(item);
        }
        Ok(())
    }

    fn changed(&mut self) {
        self.generation += 1;
        self.dirty = true;
    }

    async fn interrupt(&mut self, reason: &str) -> Result<()> {
        if let Some(inference) = self.inference.take() {
            inference.task.abort();
            self.agent
                .record_metric(RuntimeMetric::ModelRequestFinished {
                    status: MetricStatus::Aborted,
                    duration: inference.started.elapsed(),
                });
            self.append(vec![runtime(TurnItemKind::InferenceInterrupted {
                inference_id: inference.id,
                reason: reason.into(),
            })])
            .await?;
        }
        Ok(())
    }

    async fn command(&mut self, command: Command) -> Result<()> {
        match command {
            Command::Input { text, reply } => {
                let item = TurnItem::new(
                    TurnItemSource::User,
                    TurnItemKind::UserInput {
                        text,
                        response_to: None,
                    },
                );
                let id = item.id.clone();
                self.append(vec![item]).await?;
                self.changed();
                self.interrupt("superseded by new user input").await?;
                let _ = reply.send(Ok(id));
            }
            Command::Cancel {
                operation_id,
                reply,
            } => {
                let result = self.cancel(&operation_id).await;
                let _ = reply.send(result);
            }
            Command::Progress {
                operation_id,
                progress,
                reply,
            } => {
                let result = if self
                    .operations
                    .get(&operation_id)
                    .is_some_and(|op| op.abort.is_some())
                {
                    self.append(vec![TurnItem::new(
                        TurnItemSource::Tool,
                        TurnItemKind::ToolProgress {
                            operation_id,
                            progress,
                        },
                    )])
                    .await
                } else {
                    Err(AgentError::FullDuplex("operation is not running".into()))
                };
                let _ = reply.send(result);
            }
            Command::Resolve {
                operation_id,
                resolution,
                reply,
            } => {
                let result = self.resolve(&operation_id, resolution).await;
                let _ = reply.send(result);
            }
        }
        Ok(())
    }

    async fn start_inference(&mut self) -> Result<()> {
        if self.model_iterations >= self.agent.config.turn_execution_limits.max_model_iterations {
            self.begin_stop(TurnOutcome::Failed {
                error: AgentError::MaxIterations(self.model_iterations).to_string(),
            })
            .await?;
            return Ok(());
        }
        let id = new_id("inference");
        let position = self.turn().items.len();
        self.append(vec![runtime(TurnItemKind::InferenceStarted {
            inference_id: id.clone(),
            history_position: position,
        })])
        .await?;
        self.model_iterations += 1;
        self.dirty = false;
        let task = self.jobs.spawn(worker::infer(
            self.agent.clone(),
            self.thread.clone(),
            id.clone(),
            self.sender.clone(),
        ));
        self.job_ids.insert(task.id(), Job::Model(id.clone()));
        self.inference = Some(Inference {
            id,
            generation: self.generation,
            task,
            started: Instant::now(),
        });
        Ok(())
    }

    async fn worker_event(&mut self, event: WorkerEvent) -> Result<()> {
        match event {
            WorkerEvent::ModelDelta { id, event } => {
                if let ModelStreamEvent::TokenUsage { usage } = &event {
                    self.usage.add_assign(*usage);
                    Agent::apply_turn_token_usage(&mut self.thread, *usage);
                }
                if self.stopping.is_none()
                    && self
                        .inference
                        .as_ref()
                        .is_some_and(|inference| inference.id == id)
                {
                    self.emit(FullDuplexEvent::Model {
                        inference_id: id,
                        event,
                    });
                }
            }
            WorkerEvent::ModelFinished { id, response } => {
                if self.stopping.is_some()
                    || self
                        .inference
                        .as_ref()
                        .is_none_or(|inference| inference.id != id)
                {
                    return Ok(());
                }
                let inference = self.inference.take().expect("current inference");
                let generation = inference.generation;
                self.agent
                    .record_metric(RuntimeMetric::ModelRequestFinished {
                        status: if response.is_ok() {
                            MetricStatus::Completed
                        } else {
                            MetricStatus::Failed
                        },
                        duration: inference.started.elapsed(),
                    });
                match response {
                    Ok(response) => self.model_response(response, generation).await?,
                    Err(error) => {
                        self.begin_stop(TurnOutcome::Failed {
                            error: error.to_string(),
                        })
                        .await?
                    }
                }
            }
            WorkerEvent::OperationFinished {
                operation_id,
                result,
                context,
                hooks,
            } => {
                if !self.operations.contains_key(&operation_id) {
                    return Ok(());
                }
                let hook_result = match &result {
                    OperationResult::Returned(Ok(FunctionExecution::Completed { output })) => {
                        FunctionCallHookResult::Completed {
                            output: output.clone(),
                        }
                    }
                    OperationResult::Returned(Ok(FunctionExecution::SuspendedAfterExecution {
                        suspension,
                        output,
                    })) => FunctionCallHookResult::Suspended {
                        suspension: suspension.clone(),
                        output: output.clone(),
                    },
                    OperationResult::Returned(Ok(
                        FunctionExecution::SuspendedBeforeExecution { suspension },
                    )) => FunctionCallHookResult::Suspended {
                        suspension: suspension.clone(),
                        output: Value::Null,
                    },
                    OperationResult::Returned(Err(error)) => FunctionCallHookResult::Failed {
                        error: error.to_string(),
                    },
                    OperationResult::Indeterminate(reason) => FunctionCallHookResult::Failed {
                        error: reason.clone(),
                    },
                    OperationResult::AbortedBeforeExecution => FunctionCallHookResult::Failed {
                        error: "cancelled before execution".into(),
                    },
                };
                match result {
                    OperationResult::Returned(Ok(FunctionExecution::Completed { output })) => {
                        self.complete(&operation_id, ToolResult::Success { output })
                            .await?
                    }
                    OperationResult::Returned(Err(error)) => {
                        self.complete(
                            &operation_id,
                            ToolResult::Error {
                                error: error.to_string(),
                            },
                        )
                        .await?
                    }
                    OperationResult::Returned(Ok(
                        FunctionExecution::SuspendedBeforeExecution { suspension },
                    )) => self.suspend(&operation_id, suspension, true, None).await?,
                    OperationResult::Returned(Ok(FunctionExecution::SuspendedAfterExecution {
                        suspension,
                        output,
                    })) => {
                        self.suspend(&operation_id, suspension, false, Some(output))
                            .await?
                    }
                    OperationResult::Indeterminate(reason) => {
                        self.indeterminate(&operation_id, reason).await?
                    }
                    OperationResult::AbortedBeforeExecution => {
                        self.complete(
                            &operation_id,
                            ToolResult::Aborted {
                                reason: "cancelled before execution".into(),
                            },
                        )
                        .await?
                    }
                }
                if !hooks.is_empty() {
                    let agent = self.agent.clone();
                    let sender = self.sender.clone();
                    let task = self.jobs.spawn(async move {
                        let mut emit = |event| {
                            let _ = sender.try_send(WorkerEvent::Hook(event));
                        };
                        let _ = tokio::time::timeout(
                            Duration::from_secs(30),
                            agent.run_after_function_call_hooks(
                                &hooks,
                                &context,
                                hook_result,
                                &mut emit,
                            ),
                        )
                        .await;
                    });
                    self.job_ids.insert(task.id(), Job::Hook);
                }
            }
            WorkerEvent::Hook(event) => self.emit(FullDuplexEvent::Hook { event }),
        }
        Ok(())
    }

    async fn model_response(&mut self, response: ModelResponse, generation: u64) -> Result<()> {
        let (continuation, text, calls) = match response {
            ModelResponse::Assistant {
                continuation,
                text,
                function_calls,
            } => (continuation, text, function_calls),
            ModelResponse::AssistantMessage { text } => (None, Some(text), Vec::new()),
            ModelResponse::FunctionCalls { calls } => (None, None, calls),
        };
        if text.as_ref().is_none_or(String::is_empty)
            && calls.is_empty()
            && !self.agent.config.allow_empty_response
        {
            self.begin_stop(TurnOutcome::Failed {
                error: "model returned neither text nor function calls".into(),
            })
            .await?;
            return Ok(());
        }
        let mut ids = HashSet::new();
        if calls.iter().any(|call| !ids.insert(&call.call_id) || self.turn().items.iter().any(|item| matches!(&item.kind, TurnItemKind::ModelResponse { function_calls, .. } if function_calls.iter().any(|previous| previous.call_id == call.call_id)))) {
            self.begin_stop(TurnOutcome::Failed { error: "duplicate model tool-call ID".into() }).await?;
            return Ok(());
        }
        let mut items = vec![TurnItem::new(
            TurnItemSource::Model,
            TurnItemKind::ModelResponse {
                continuation,
                text,
                function_calls: calls.clone(),
            },
        )];
        let mut starts = Vec::new();
        let mut closing = false;
        let mut limit_exceeded = false;
        let mut continue_after_receipts = false;
        let mut cancellations = Vec::new();
        for call in &calls {
            if self.function_calls >= self.agent.config.turn_execution_limits.max_function_calls {
                items.push(tool_output(
                    call,
                    Err("turn function-call limit exceeded".into()),
                ));
                limit_exceeded = true;
                continue;
            }
            self.function_calls += 1;
            match call.name.as_str() {
                "end_turn" => {
                    let can_close = calls.len() == 1
                        && self.operations.is_empty()
                        && generation == self.generation
                        && self.input.is_empty();
                    let result = if can_close
                        && call
                            .arguments
                            .as_object()
                            .is_some_and(|args| args.is_empty())
                    {
                        closing = true;
                        Ok(serde_json::json!({"status":"closed"}))
                    } else {
                        continue_after_receipts = true;
                        Err("cannot close: active operations, unseen events, pending input, invalid arguments, or other calls in this batch".into())
                    };
                    items.push(tool_output(call, result));
                }
                "cancel_operation" => {
                    let result = if let Some(id) =
                        call.arguments.get("operation_id").and_then(Value::as_str)
                    {
                        if self
                            .operations
                            .get(id)
                            .is_some_and(|operation| operation.abort.is_some())
                        {
                            cancellations.push(id.to_string());
                            Ok(
                                serde_json::json!({"status":"cancellation_requested", "operation_id":id}),
                            )
                        } else {
                            Err("operation is not running".into())
                        }
                    } else {
                        Err("operation_id is required".into())
                    };
                    items.push(tool_output(call, result));
                    continue_after_receipts = true;
                }
                _ => {
                    let registered = self
                        .agent
                        .function_registry
                        .recovery_policy(&call.name)
                        .is_ok();
                    if !registered
                        || self.operations.len() + starts.len()
                            >= self.config.max_concurrent_operations
                    {
                        items.push(tool_output(
                            call,
                            Err(if registered {
                                "operation capacity is full".into()
                            } else {
                                format!("unknown function: {}", call.name)
                            }),
                        ));
                        continue_after_receipts = true;
                        continue;
                    }
                    let operation_id = new_id("operation");
                    items.push(runtime(TurnItemKind::ToolAccepted {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        operation_id: operation_id.clone(),
                    }));
                    items.push(runtime(TurnItemKind::FunctionCallStarted {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        attempt: 1,
                    }));
                    starts.push((operation_id, call.clone()));
                }
            }
        }
        // One commit publishes the whole response and every protocol receipt.
        // No accepted external operation starts before that commit succeeds.
        for id in &cancellations {
            items.push(runtime(TurnItemKind::ToolCancellationRequested {
                operation_id: id.clone(),
            }));
        }
        if closing {
            // Seal ingress synchronously with the closure check. Submissions
            // racing the durable commit receive a closed error, never a false ack.
            self.input.close();
            Agent::set_turn_status(&mut self.thread, &self.turn_id, TurnStatus::Completed)?;
        }
        self.append(items).await?;
        for id in cancellations {
            if let Some(abort) = self
                .operations
                .get(&id)
                .and_then(|operation| operation.abort.as_ref())
            {
                abort.abort();
            }
        }
        for (id, call) in starts {
            self.launch(id, call, 1);
        }
        if closing {
            self.stopping = Some(TurnOutcome::CompletedWithoutMessage);
        } else if limit_exceeded {
            // Calls admitted within budget still finish. Excess calls already
            // have durable error receipts in this same batch.
            self.stopping = Some(TurnOutcome::Failed {
                error: AgentError::MaxFunctionCalls(self.function_calls).to_string(),
            });
            self.dirty = false;
        } else if continue_after_receipts {
            self.changed();
        } else if !self.dirty {
            self.emit(FullDuplexEvent::Idle);
        }
        Ok(())
    }

    fn launch(&mut self, id: String, call: ModelFunctionCall, attempt: u32) {
        let (abort, signal) = turn_abort_pair();
        let context = FunctionContext {
            thread_id: self.thread.id.clone(),
            metadata: self.thread.metadata.clone(),
            turn_id: self.turn_id.clone(),
            call_id: call.call_id.clone(),
            projection: ThreadProjection::from_thread(&self.thread),
            abort_signal: signal,
        };
        let task = self.jobs.spawn(worker::execute(
            self.agent.clone(),
            call.clone(),
            id.clone(),
            context,
            self.config.cancellation_grace,
            self.sender.clone(),
        ));
        self.job_ids.insert(task.id(), Job::Operation(id.clone()));
        self.operations.insert(
            id,
            Operation {
                call,
                abort: Some(abort),
                suspension: None,
                attempt,
            },
        );
    }

    async fn complete(&mut self, id: &str, result: ToolResult) -> Result<()> {
        let op = self
            .operations
            .get(id)
            .ok_or_else(|| AgentError::FullDuplex("unknown operation".into()))?;
        self.append(vec![TurnItem::new(
            TurnItemSource::Tool,
            TurnItemKind::ToolCompleted {
                call_id: op.call.call_id.clone(),
                name: op.call.name.clone(),
                operation_id: id.into(),
                result,
            },
        )])
        .await?;
        self.operations.remove(id);
        self.changed();
        Ok(())
    }

    async fn cancel(&mut self, id: &str) -> Result<()> {
        let abort = self
            .operations
            .get(id)
            .and_then(|op| op.abort.clone())
            .ok_or_else(|| AgentError::FullDuplex("operation is not running".into()))?;
        self.append(vec![runtime(TurnItemKind::ToolCancellationRequested {
            operation_id: id.into(),
        })])
        .await?;
        abort.abort();
        Ok(())
    }

    async fn indeterminate(&mut self, id: &str, reason: String) -> Result<()> {
        self.append(vec![runtime(TurnItemKind::ToolIndeterminate {
            operation_id: id.into(),
            reason,
        })])
        .await?;
        if let Some(operation) = self.operations.get_mut(id) {
            operation.abort = None;
        }
        self.changed();
        Ok(())
    }

    async fn suspend(
        &mut self,
        id: &str,
        suspension: Suspension,
        before: bool,
        output: Option<Value>,
    ) -> Result<()> {
        self.append(vec![runtime(TurnItemKind::ToolSuspended {
            operation_id: id.into(),
            suspension: suspension.clone(),
            before_execution: before,
            output: output.clone(),
        })])
        .await?;
        if let Some(operation) = self.operations.get_mut(id) {
            operation.abort = None;
            operation.suspension = Some((suspension, before, output));
        }
        self.changed();
        Ok(())
    }

    async fn resolve(&mut self, id: &str, resolution: SuspensionResolution) -> Result<()> {
        let op = self
            .operations
            .get(id)
            .ok_or_else(|| AgentError::FullDuplex("unknown operation".into()))?;
        if op.abort.is_some() {
            return Err(AgentError::FullDuplex(
                "cannot reconcile a running operation".into(),
            ));
        }
        match resolution {
            SuspensionResolution::ExternalResult { output } => {
                self.complete(id, ToolResult::Success { output }).await
            }
            SuspensionResolution::Deny { reason } => {
                self.complete(id, ToolResult::Error { error: reason }).await
            }
            SuspensionResolution::UserInput { text } => {
                let (suspension, before, output) = op
                    .suspension
                    .clone()
                    .ok_or_else(|| AgentError::FullDuplex("operation is not suspended".into()))?;
                if before {
                    return Err(AgentError::FullDuplex(
                        "pre-execution suspension requires approval or denial".into(),
                    ));
                }
                self.append(vec![TurnItem::new(
                    TurnItemSource::User,
                    TurnItemKind::UserInput {
                        text,
                        response_to: Some(suspension.id),
                    },
                )])
                .await?;
                self.complete(
                    id,
                    ToolResult::Success {
                        output: output.unwrap_or(Value::Null),
                    },
                )
                .await?;
                self.interrupt("suspension resolved by user input").await
            }
            SuspensionResolution::Approve => {
                if op
                    .suspension
                    .as_ref()
                    .is_some_and(|(_, before, _)| !*before)
                {
                    return Err(AgentError::FullDuplex(
                        "post-execution suspension requires external result or user input".into(),
                    ));
                }
                let call = op.call.clone();
                let attempt = op.attempt + 1;
                self.append(vec![
                    runtime(TurnItemKind::ToolResumed {
                        operation_id: id.into(),
                    }),
                    runtime(TurnItemKind::FunctionCallStarted {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        attempt,
                    }),
                ])
                .await?;
                self.launch(id.into(), call, attempt);
                Ok(())
            }
        }
    }

    async fn recover_operations(&mut self) -> Result<()> {
        self.trace_sequence = self.turn().items.len() as u64;
        let mut unfinished_inference = None;
        for item in &self.turn().items {
            match &item.kind {
                TurnItemKind::InferenceStarted { inference_id, .. } => {
                    unfinished_inference = Some(inference_id.clone())
                }
                TurnItemKind::InferenceInterrupted { .. } | TurnItemKind::ModelResponse { .. } => {
                    unfinished_inference = None
                }
                _ => {}
            }
        }
        if let Some(inference_id) = unfinished_inference {
            self.append(vec![runtime(TurnItemKind::InferenceInterrupted {
                inference_id,
                reason: "previous process ended before inference completed".into(),
            })])
            .await?;
        }
        self.model_iterations = self
            .turn()
            .items
            .iter()
            .filter(|item| matches!(item.kind, TurnItemKind::InferenceStarted { .. }))
            .count();
        self.function_calls = self
            .turn()
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                TurnItemKind::ModelResponse { function_calls, .. } => Some(function_calls.len()),
                _ => None,
            })
            .sum();
        let projection = ThreadProjection::from_thread(&self.thread);
        for (id, operation) in projection.active_operations {
            let call = self
                .turn()
                .items
                .iter()
                .find_map(|item| match &item.kind {
                    TurnItemKind::ModelResponse { function_calls, .. } => function_calls
                        .iter()
                        .find(|call| call.call_id == operation.call_id)
                        .cloned(),
                    _ => None,
                })
                .ok_or_else(|| {
                    AgentError::FullDuplex("operation has no originating call".into())
                })?;
            let attempt = self
                .turn()
                .items
                .iter()
                .filter_map(|item| match &item.kind {
                    TurnItemKind::FunctionCallStarted {
                        call_id, attempt, ..
                    } if *call_id == call.call_id => Some(*attempt),
                    _ => None,
                })
                .max()
                .unwrap_or(0);
            let suspension = self
                .turn()
                .items
                .iter()
                .rev()
                .find_map(|item| match &item.kind {
                    TurnItemKind::ToolSuspended {
                        operation_id,
                        suspension,
                        before_execution,
                        output,
                    } if *operation_id == id => {
                        Some((suspension.clone(), *before_execution, output.clone()))
                    }
                    _ => None,
                });
            let retry = operation.status == OperationStatus::Running
                && self
                    .agent
                    .function_registry
                    .recovery_policy(&call.name)
                    .unwrap_or(FunctionRecoveryPolicy::NonIdempotent)
                    == FunctionRecoveryPolicy::Idempotent;
            self.operations.insert(
                id.clone(),
                Operation {
                    call: call.clone(),
                    abort: None,
                    suspension: if operation.status == OperationStatus::Suspended {
                        suspension
                    } else {
                        None
                    },
                    attempt,
                },
            );
            if retry {
                self.append(vec![
                    runtime(TurnItemKind::ToolResumed {
                        operation_id: id.clone(),
                    }),
                    runtime(TurnItemKind::FunctionCallStarted {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        attempt: attempt + 1,
                    }),
                ])
                .await?;
                self.launch(id, call, attempt + 1);
            } else if operation.status != OperationStatus::Suspended {
                self.indeterminate(
                    &id,
                    "unfinished operation recovered; reconcile before retrying".into(),
                )
                .await?;
            }
        }
        Ok(())
    }

    async fn begin_stop(&mut self, outcome: TurnOutcome) -> Result<()> {
        self.interrupt("interaction stopping").await?;
        self.stopping = Some(outcome);
        self.dirty = false;
        let ids: Vec<_> = self
            .operations
            .iter()
            .filter(|(_, op)| op.abort.is_some())
            .map(|(id, _)| id.clone())
            .collect();
        for id in ids {
            self.cancel(&id).await?;
        }
        Ok(())
    }

    async fn finish(&mut self) -> Result<TurnOutcome> {
        let outcome = self.stopping.take().expect("stopping outcome");
        let (status, trace_status, items) = match &outcome {
            TurnOutcome::Aborted { reason } => (
                TurnStatus::Aborted,
                TraceTurnStatus::Aborted,
                vec![runtime(TurnItemKind::TurnAborted {
                    reason: reason.clone(),
                })],
            ),
            TurnOutcome::Failed { error } => (
                TurnStatus::Failed,
                TraceTurnStatus::Failed,
                vec![runtime(TurnItemKind::TurnFailed {
                    error: error.clone(),
                })],
            ),
            _ => (
                TurnStatus::Completed,
                TraceTurnStatus::Completed,
                Vec::new(),
            ),
        };
        Agent::set_turn_status(&mut self.thread, &self.turn_id, status)?;
        self.append(items).await?;
        self.agent.record_trace(
            &mut self.trace_sequence,
            &self.thread.id,
            &self.turn_id,
            TraceEventKind::TurnFinished {
                status: trace_status,
            },
        );
        self.agent.record_metric(RuntimeMetric::TokenUsage {
            input_tokens: self.usage.input_tokens,
            cached_input_tokens: self.usage.cached_input_tokens,
            output_tokens: self.usage.output_tokens,
            total_tokens: self.usage.total_tokens,
        });
        Ok(outcome)
    }
}

fn runtime(kind: TurnItemKind) -> TurnItem {
    TurnItem::new(TurnItemSource::Runtime, kind)
}

fn tool_output(call: &ModelFunctionCall, result: std::result::Result<Value, String>) -> TurnItem {
    TurnItem::new(
        TurnItemSource::Tool,
        TurnItemKind::ToolOutput {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            result: match result {
                Ok(output) => ToolResult::Success { output },
                Err(error) => ToolResult::Error { error },
            },
        },
    )
}
