use super::worker::{self, OperationResult, WorkerEvent};
use super::ModelIngress;
use super::{Command, TurnConfig, TurnEvent};
use crate::agent::ActiveTurnGuard;
use crate::metrics::{MetricStatus, RuntimeMetric};
use crate::trace::{TraceEventKind, TraceTurnStatus};
use crate::{
    turn_abort_pair, Agent, AgentError, FunctionCallHookResult, FunctionContext, FunctionExecution,
    FunctionRecoveryPolicy, ModelFunctionCall, ModelResponse, ModelStreamEvent, Result,
    SessionLease, SuspensionResolution, TurnAbortHandle, TurnOutcome,
};
use crate::{ModelInteraction, ModelSessionCommand};
use crate::{ModelResponseStatus, ModelSession, ModelSessionConfig, ModelSessionEvent};
use canary_agent_kernel::projection::OperationStatus;
use canary_agent_kernel::SuspensionKind;
use canary_agent_kernel::{
    new_id, Suspension, Thread, ThreadProjection, TokenUsage, ToolResult, Turn, TurnItem,
    TurnItemKind, TurnItemSource, TurnStatus,
};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tokio::sync::{broadcast, mpsc};
use tokio::task::{AbortHandle, Id, JoinSet};

struct Connection {
    id: String,
    shutdown: Option<oneshot::Sender<()>>,
    responses: HashMap<String, Response>,
    active_inputs: HashSet<String>,
    known_inputs: HashSet<String>,
    committed_inputs: HashSet<String>,
    transcripts: HashSet<String>,
    completed_items: HashSet<String>,
    usage_events: HashSet<String>,
    ready: bool,
    history_position: usize,
}

struct Response {
    generation: u64,
    finished: bool,
    interrupted: bool,
}

struct Inference {
    id: String,
    generation: u64,
    task: AbortHandle,
    started: Instant,
}

struct ModelIteration {
    continuation: Option<crate::ModelContinuation>,
    text: Option<String>,
    calls: Vec<ModelFunctionCall>,
}

impl From<ModelResponse> for ModelIteration {
    fn from(response: ModelResponse) -> Self {
        match response {
            ModelResponse::Assistant {
                continuation,
                text,
                function_calls,
            } => Self {
                continuation,
                text,
                calls: function_calls,
            },
            ModelResponse::AssistantMessage { text } => Self {
                continuation: None,
                text: Some(text),
                calls: Vec::new(),
            },
            ModelResponse::FunctionCalls { calls } => Self {
                continuation: None,
                text: None,
                calls,
            },
        }
    }
}

struct OperationStart {
    operation_id: String,
    call: ModelFunctionCall,
    attempt: u32,
}

#[derive(Clone)]
struct OperationSuspension {
    suspension: Suspension,
    before_execution: bool,
    output: Option<Value>,
}

struct Operation {
    call: ModelFunctionCall,
    abort: Option<TurnAbortHandle>,
    suspension: Option<OperationSuspension>,
    attempt: u32,
}

enum Job {
    Connection(String),
    Model(String),
    Operation(String),
    Hook,
}

pub(super) struct Runner {
    agent: Agent,
    thread: Thread,
    turn_id: String,
    config: TurnConfig,
    input: mpsc::Receiver<Command>,
    output: broadcast::Sender<TurnEvent>,
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
    model_input: ModelIngress,
    connection: Option<Connection>,
    connection_error: Option<String>,
    ending_response: Option<String>,
}

impl Runner {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        agent: Agent,
        thread: Thread,
        turn_id: String,
        config: TurnConfig,
        input: mpsc::Receiver<Command>,
        output: broadcast::Sender<TurnEvent>,
        guard: ActiveTurnGuard,
        lease: SessionLease,
        model_input: ModelIngress,
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
            model_input,
            connection: None,
            connection_error: None,
            ending_response: None,
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
        *self.model_input.lock().expect("model ingress") = None;
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
            self.emit(TurnEvent::Finished {
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
        if self.bidirectional() {
            self.connect_model().await?;
        }
        loop {
            if let Some(reason) = self.connection_error.take() {
                if let Some(id) = self.connection.as_ref().map(|c| c.id.clone()) {
                    self.connection_lost(&id, reason).await?;
                }
            }
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
                if !self.bidirectional() && self.inference.is_none() && self.dirty {
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
                                    Some(Job::Connection(id)) => self.connection_lost(&id, format!("connection worker failed: {error}")).await?,
                                    Some(Job::Operation(operation_id)) => self.indeterminate(&operation_id, format!("worker failed: {error}")).await?,
                                    Some(Job::Model(id)) if self.inference.as_ref().is_some_and(|current| current.id == id) => {
                                        self.begin_stop(TurnOutcome::Failed { error: format!("model worker failed: {error}") }).await?;
                                    }
                                    _ => tracing::warn!(%error, "turn auxiliary worker failed"),
                                }
                            }
                        }
                        None => {}
                    }
                }
            }
        }
    }

    fn bidirectional(&self) -> bool {
        self.agent.model_client.interaction() == ModelInteraction::Bidirectional
    }

    async fn connect_model(&mut self) -> Result<()> {
        let id = new_id("connection");
        let (sender, receiver) = mpsc::channel(self.config.input_capacity);
        // Input is admitted only after the connection-open fact has committed.
        *self.model_input.lock().expect("model ingress") = None;
        let (shutdown, stop) = oneshot::channel();
        self.connection = Some(Connection {
            id: id.clone(),
            shutdown: Some(shutdown),
            responses: HashMap::new(),
            active_inputs: HashSet::new(),
            known_inputs: HashSet::new(),
            committed_inputs: HashSet::new(),
            transcripts: HashSet::new(),
            completed_items: HashSet::new(),
            usage_events: HashSet::new(),
            ready: false,
            history_position: self.turn().items.len(),
        });
        let task = self.jobs.spawn(drive_session(
            self.agent.clone(),
            self.thread.clone(),
            self.turn_id.clone(),
            id.clone(),
            receiver,
            sender,
            stop,
            self.config.cancellation_grace,
            self.sender.clone(),
        ));
        self.job_ids.insert(task.id(), Job::Connection(id));
        Ok(())
    }

    async fn connection_opened(
        &mut self,
        id: &str,
        ingress: mpsc::Sender<ModelSessionCommand>,
    ) -> Result<()> {
        if !self.is_current_connection(id) || self.stopping.is_some() {
            return Ok(());
        }
        Agent::set_turn_status(&mut self.thread, &self.turn_id, TurnStatus::Running)?;
        self.append(vec![runtime(TurnItemKind::ModelSessionOpened {
            connection_id: id.into(),
        })])
        .await?;
        self.connection.as_mut().expect("connection").ready = true;
        *self.model_input.lock().expect("model ingress") = Some(ingress);
        let position = self
            .connection
            .as_ref()
            .expect("connection")
            .history_position;
        let catch_up = self.turn().items[position..].to_vec();
        self.notify_model(&catch_up);
        self.emit(TurnEvent::SessionReady {
            connection_id: id.into(),
        });
        Ok(())
    }

    fn is_current_connection(&self, id: &str) -> bool {
        self.connection
            .as_ref()
            .is_some_and(|connection| connection.id == id)
    }

    fn shutdown_connection(&mut self) {
        *self.model_input.lock().expect("model ingress") = None;
        if let Some(mut connection) = self.connection.take() {
            if let Some(shutdown) = connection.shutdown.take() {
                let _ = shutdown.send(());
            }
        }
    }

    async fn connection_lost(&mut self, id: &str, reason: String) -> Result<()> {
        if !self.is_current_connection(id) || self.stopping.is_some() {
            return Ok(());
        }
        self.shutdown_connection();
        let suspension = Suspension {
            id: new_id("suspension"),
            kind: SuspensionKind::ModelConnection,
            payload: serde_json::json!({"connection_id":id,"reason":reason}),
        };
        Agent::set_turn_status(&mut self.thread, &self.turn_id, TurnStatus::Suspended)?;
        self.append(vec![runtime(TurnItemKind::ModelSessionSuspended {
            connection_id: id.into(),
            suspension: suspension.clone(),
        })])
        .await?;
        self.emit(TurnEvent::Suspended { suspension });
        Ok(())
    }

    fn notify_model(&mut self, items: &[TurnItem]) {
        if !self.bidirectional() {
            return;
        }
        let items: Vec<_> = items
            .iter()
            .filter(|item| {
                matches!(
                    item.kind,
                    TurnItemKind::ToolAccepted { .. }
                        | TurnItemKind::ToolCompleted { .. }
                        | TurnItemKind::ToolOutput { .. }
                        | TurnItemKind::ToolProgress { .. }
                        | TurnItemKind::ToolSuspended { .. }
                        | TurnItemKind::ToolIndeterminate { .. }
                        | TurnItemKind::ToolCancellationRequested { .. }
                        | TurnItemKind::ToolResumed { .. }
                        | TurnItemKind::UserInput {
                            response_to: Some(_),
                            ..
                        }
                )
            })
            .cloned()
            .collect();
        if items.is_empty() {
            return;
        }
        if let Some(sender) = self.model_input.lock().expect("model ingress").as_ref() {
            if let Err(error) = sender.try_send(ModelSessionCommand::Observe { items }) {
                self.connection_error = Some(format!("model observation delivery failed: {error}"));
            }
        }
    }

    fn session_can_close(&self) -> bool {
        !self.bidirectional()
            || self
                .connection
                .as_ref()
                .is_some_and(|connection| connection.ready && connection.active_inputs.is_empty())
    }

    async fn session_event(&mut self, id: &str, event: ModelSessionEvent) -> Result<()> {
        if !self.is_current_connection(id) {
            return Ok(());
        }
        if self.stopping.is_some() {
            let draining = self
                .ending_response
                .as_ref()
                .is_some_and(|ending| match &event {
                    ModelSessionEvent::TextDelta { response_id, .. }
                    | ModelSessionEvent::AudioDelta { response_id, .. }
                    | ModelSessionEvent::ResponseFinished { response_id, .. } => {
                        response_id == ending
                    }
                    ModelSessionEvent::Usage { .. } => true,
                    ModelSessionEvent::TranscriptFinalized { .. } => true,
                    _ => false,
                });
            if !draining {
                return Ok(());
            }
        }
        let connection = self.connection.as_mut().expect("connection");
        let mut facts = Vec::new();
        match &event {
            ModelSessionEvent::InputStarted { item_id } => {
                if !connection.known_inputs.insert(item_id.clone()) {
                    return Ok(());
                }
                connection.active_inputs.insert(item_id.clone());
                facts.push(runtime(TurnItemKind::ModelInputStarted {
                    connection_id: id.into(),
                    item_id: item_id.clone(),
                }));
                self.changed();
            }
            ModelSessionEvent::InputEnded { item_id } => {
                facts.push(runtime(TurnItemKind::ModelInputEnded {
                    connection_id: id.into(),
                    item_id: item_id.clone(),
                }));
            }
            ModelSessionEvent::InputCommitted { item_id } => {
                if !connection.committed_inputs.insert(item_id.clone()) {
                    return Ok(());
                }
                connection.active_inputs.remove(item_id);
                let new = connection.known_inputs.insert(item_id.clone());
                facts.push(runtime(TurnItemKind::ModelInputCommitted {
                    connection_id: id.into(),
                    item_id: item_id.clone(),
                }));
                if new {
                    self.changed();
                }
            }
            ModelSessionEvent::TranscriptFinalized { item_id, text } => {
                if !connection.transcripts.insert(item_id.clone()) {
                    return Ok(());
                }
                facts.push(TurnItem::new(
                    TurnItemSource::User,
                    TurnItemKind::InputTranscript {
                        connection_id: id.into(),
                        item_id: item_id.clone(),
                        text: text.clone(),
                    },
                ));
            }
            ModelSessionEvent::ResponseStarted { response_id } => {
                if connection.responses.contains_key(response_id) {
                    return Ok(());
                }
                if self.model_iterations
                    >= self.agent.config.turn_execution_limits.max_model_iterations
                {
                    self.begin_stop(TurnOutcome::Failed {
                        error: AgentError::MaxIterations(self.model_iterations).to_string(),
                    })
                    .await?;
                    return Ok(());
                }
                connection.responses.insert(
                    response_id.clone(),
                    Response {
                        generation: self.generation,
                        finished: false,
                        interrupted: false,
                    },
                );
                self.model_iterations += 1;
                facts.push(runtime(TurnItemKind::ModelResponseStarted {
                    connection_id: id.into(),
                    response_id: response_id.clone(),
                }));
            }
            ModelSessionEvent::TextDelta { response_id, .. }
            | ModelSessionEvent::AudioDelta { response_id, .. } => {
                if connection
                    .responses
                    .get(response_id)
                    .is_none_or(|r| r.finished)
                {
                    return Ok(());
                }
            }
            ModelSessionEvent::ItemCompleted {
                response_id,
                item_id,
                text,
                function_calls,
            } => {
                let Some(response) = connection.responses.get(response_id) else {
                    self.connection_lost(id, "output item has no response-start event".into())
                        .await?;
                    return Ok(());
                };
                if response.finished
                    || response.interrupted
                    || !connection.completed_items.insert(item_id.clone())
                {
                    return Ok(());
                }
                let generation = response.generation;
                let origin = runtime(TurnItemKind::ModelItemReceived {
                    connection_id: id.into(),
                    response_id: response_id.clone(),
                    item_id: item_id.clone(),
                });
                self.model_response(
                    ModelResponse::Assistant {
                        continuation: None,
                        text: text.clone(),
                        function_calls: function_calls.clone(),
                    },
                    generation,
                    Some(origin),
                )
                .await?;
            }
            ModelSessionEvent::ResponseFinished {
                response_id,
                status,
            } => {
                let Some(response) = connection.responses.get_mut(response_id) else {
                    return Ok(());
                };
                if response.finished {
                    return Ok(());
                }
                response.finished = true;
                response.interrupted = matches!(status, ModelResponseStatus::Interrupted { .. });
                facts.push(runtime(TurnItemKind::ModelResponseFinished {
                    connection_id: id.into(),
                    response_id: response_id.clone(),
                    status: status.clone(),
                }));
                if let ModelResponseStatus::Failed { error } = status {
                    self.append(facts).await?;
                    self.begin_stop(TurnOutcome::Failed {
                        error: error.clone(),
                    })
                    .await?;
                    return Ok(());
                }
                if self.ending_response.as_deref() == Some(response_id) {
                    self.shutdown_connection();
                }
            }
            ModelSessionEvent::Usage {
                event_id,
                response_id,
                usage,
            } => {
                if !connection.usage_events.insert(event_id.clone()) {
                    return Ok(());
                }
                self.usage.add_assign(*usage);
                Agent::apply_turn_token_usage(&mut self.thread, *usage);
                facts.push(runtime(TurnItemKind::ModelUsageReported {
                    connection_id: id.into(),
                    event_id: event_id.clone(),
                    response_id: response_id.clone(),
                    usage: *usage,
                }));
            }
        }
        if !facts.is_empty() {
            self.append(facts).await?;
        }
        self.emit(TurnEvent::Session {
            connection_id: id.into(),
            event,
        });
        Ok(())
    }

    fn turn(&self) -> &Turn {
        self.thread
            .turns
            .iter()
            .find(|turn| turn.id == self.turn_id)
            .expect("owned turn")
    }

    fn emit(&self, event: TurnEvent) {
        let _ = self.output.send(event);
    }

    fn publish_item(&mut self, item: TurnItem) {
        self.agent.record_trace(
            &mut self.trace_sequence,
            &self.thread.id,
            &self.turn_id,
            TraceEventKind::InteractionItem { item: item.clone() },
        );
        self.emit(TurnEvent::ItemCommitted { item });
    }

    async fn append(&mut self, items: Vec<TurnItem>) -> Result<()> {
        let mut thread = self.thread.clone();
        Agent::push_turn_items(&mut thread, &self.turn_id, items.clone())?;
        self.thread = self.agent.commit_thread(thread, self.lease.fence()).await?;
        self.notify_model(&items);
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
            Command::Reconnect { reply } => {
                let result = if !self.bidirectional() {
                    Err(AgentError::Model(
                        "request-based model has no persistent connection".into(),
                    ))
                } else if self.connection.is_some() {
                    Err(AgentError::Model(
                        "model connection is already active or connecting".into(),
                    ))
                } else {
                    self.connect_model().await
                };
                let _ = reply.send(result);
            }
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
                    Err(AgentError::TurnExecution("operation is not running".into()))
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
            WorkerEvent::SessionOpened {
                connection_id,
                ingress,
            } => self.connection_opened(&connection_id, ingress).await?,
            WorkerEvent::SessionEvent {
                connection_id,
                event,
            } => self.session_event(&connection_id, event).await?,
            WorkerEvent::SessionDisconnected {
                connection_id,
                reason,
            } => self.connection_lost(&connection_id, reason).await?,
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
                    self.emit(TurnEvent::Model {
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
                    Ok(response) => self.model_response(response, generation, None).await?,
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
            WorkerEvent::Hook(event) => self.emit(TurnEvent::Hook { event }),
        }
        Ok(())
    }

    async fn model_response(
        &mut self,
        response: ModelResponse,
        generation: u64,
        origin: Option<TurnItem>,
    ) -> Result<()> {
        let response_id = origin.as_ref().and_then(|item| match &item.kind {
            TurnItemKind::ModelItemReceived { response_id, .. } => Some(response_id.clone()),
            _ => None,
        });
        let ModelIteration {
            continuation,
            text,
            calls,
        } = response.into();
        if self.bidirectional() && text.as_ref().is_none_or(String::is_empty) && calls.is_empty() {
            // An audio-only item is still a fact, but it is not an empty text
            // assistant message in a future Chat Completions projection.
            self.append(origin.into_iter().collect()).await?;
            return Ok(());
        }
        if text.as_ref().is_none_or(String::is_empty)
            && calls.is_empty()
            && !self.agent.config.allow_empty_response
            && !self.bidirectional()
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
        let mut items: Vec<_> = origin.into_iter().collect();
        items.push(TurnItem::new(
            TurnItemSource::Model,
            TurnItemKind::ModelResponse {
                continuation,
                text,
                function_calls: calls.clone(),
            },
        ));
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
                    let can_close = can_close && self.session_can_close();
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
                    starts.push(OperationStart {
                        operation_id,
                        call: call.clone(),
                        attempt: 1,
                    });
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
        for start in starts {
            self.launch(start);
        }
        if closing {
            self.stopping = Some(TurnOutcome::Completed);
            self.ending_response = response_id;
            *self.model_input.lock().expect("model ingress") = None;
            if self.ending_response.is_none() {
                self.shutdown_connection();
            }
        } else if limit_exceeded {
            // Calls admitted within budget still finish. Excess calls already
            // have durable error receipts in this same batch.
            self.stopping = Some(TurnOutcome::Failed {
                error: AgentError::MaxFunctionCalls(self.function_calls).to_string(),
            });
            self.dirty = false;
            self.shutdown_connection();
        } else if continue_after_receipts {
            self.changed();
        } else if !self.dirty && !self.bidirectional() {
            self.emit(TurnEvent::Idle);
        }
        Ok(())
    }

    fn launch(&mut self, start: OperationStart) {
        let OperationStart {
            operation_id: id,
            call,
            attempt,
        } = start;
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
            .ok_or_else(|| AgentError::TurnExecution("unknown operation".into()))?;
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
            .ok_or_else(|| AgentError::TurnExecution("operation is not running".into()))?;
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
            operation.suspension = Some(OperationSuspension {
                suspension,
                before_execution: before,
                output,
            });
        }
        self.changed();
        Ok(())
    }

    async fn resolve(&mut self, id: &str, resolution: SuspensionResolution) -> Result<()> {
        let op = self
            .operations
            .get(id)
            .ok_or_else(|| AgentError::TurnExecution("unknown operation".into()))?;
        if op.abort.is_some() {
            return Err(AgentError::TurnExecution(
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
                let OperationSuspension {
                    suspension,
                    before_execution,
                    output,
                } = op.suspension.clone().ok_or_else(|| {
                    AgentError::TurnExecution("operation is not suspended".into())
                })?;
                if before_execution {
                    return Err(AgentError::TurnExecution(
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
                    .is_some_and(|state| !state.before_execution)
                {
                    return Err(AgentError::TurnExecution(
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
                self.launch(OperationStart {
                    operation_id: id.into(),
                    call,
                    attempt,
                });
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
            .filter(|item| {
                matches!(
                    item.kind,
                    TurnItemKind::InferenceStarted { .. }
                        | TurnItemKind::ModelResponseStarted { .. }
                )
            })
            .count();
        let recovered_usage: Vec<_> = self
            .turn()
            .items
            .iter()
            .filter_map(|item| match item.kind {
                TurnItemKind::ModelUsageReported { usage, .. } => Some(usage),
                _ => None,
            })
            .collect();
        for usage in recovered_usage {
            self.usage.add_assign(usage);
        }
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
                    AgentError::TurnExecution("operation has no originating call".into())
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
                    } if *operation_id == id => Some(OperationSuspension {
                        suspension: suspension.clone(),
                        before_execution: *before_execution,
                        output: output.clone(),
                    }),
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
                self.launch(OperationStart {
                    operation_id: id,
                    call,
                    attempt: attempt + 1,
                });
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
        self.shutdown_connection();
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

#[allow(clippy::too_many_arguments)]
async fn drive_session(
    agent: Agent,
    history: Thread,
    turn_id: String,
    connection_id: String,
    mut commands: mpsc::Receiver<ModelSessionCommand>,
    ingress: mpsc::Sender<ModelSessionCommand>,
    mut stop: oneshot::Receiver<()>,
    grace: Duration,
    events: mpsc::Sender<WorkerEvent>,
) {
    let connect = async {
        let request = worker::prepare_request(&agent, &history).await?;
        agent
            .model_client
            .connect(ModelSessionConfig {
                connection_id: connection_id.clone(),
                turn_id,
                request,
                history,
                model: agent.model_client.model_descriptor(),
            })
            .await
    };
    let connected = tokio::select! {
        biased;
        _ = &mut stop => return,
        result = connect => result
    };
    let result = match connected {
        Ok(ModelSession {
            mut sender,
            mut receiver,
        }) => {
            // Sealing public ingress must not close the writer before the final
            // response's audio/usage has drained.
            let _input_owner = ingress.clone();
            // Reader and writer run independently, including while send awaits IO.
            if events
                .send(WorkerEvent::SessionOpened {
                    connection_id: connection_id.clone(),
                    ingress,
                })
                .await
                .is_err()
            {
                return;
            }
            let result = {
                let read = async {
                    loop {
                        match receiver.receive().await? {
                            Some(event) => events
                                .send(WorkerEvent::SessionEvent {
                                    connection_id: connection_id.clone(),
                                    event,
                                })
                                .await
                                .map_err(|_| AgentError::Model("runtime closed".into()))?,
                            None => {
                                return Err::<(), _>(AgentError::Model(
                                    "model connection closed".into(),
                                ))
                            }
                        }
                    }
                };
                let write = async {
                    while let Some(command) = commands.recv().await {
                        sender.send(command).await?;
                    }
                    Err::<(), _>(AgentError::Model("model input closed".into()))
                };
                tokio::select! {
                    biased;
                    _ = &mut stop => Ok(()),
                    result = async { tokio::try_join!(read, write).map(|_| ()) } => result
                }
            };
            let _ = tokio::time::timeout(grace, sender.send(ModelSessionCommand::Close)).await;
            result
        }
        Err(error) => Err(error),
    };
    if let Err(error) = result {
        let _ = events
            .send(WorkerEvent::SessionDisconnected {
                connection_id,
                reason: error.to_string(),
            })
            .await;
    }
}
