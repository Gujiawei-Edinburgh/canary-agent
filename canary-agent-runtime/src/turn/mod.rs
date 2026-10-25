//! Open interactions with interruptible inference and independently tracked tools.
mod runner;
mod worker;

use crate::functions::SuspensionResolution;
use crate::model::ModelStreamEvent;
use crate::{Agent, AgentError, FunctionHookEvent, Result, TurnAbortHandle, TurnOutcome};
use crate::{ModelInput, ModelInteraction, ModelSessionCommand, ModelSessionEvent};
use canary_agent_kernel::Suspension;
use canary_agent_kernel::{Thread, ThreadProjection, TurnStatus};
use canary_agent_kernel::{Turn, TurnItem, TurnItemKind, TurnItemSource};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::JoinHandle;

#[derive(Debug, Clone)]
/// Queue capacity, operation concurrency, and cancellation settings for a turn.
pub struct TurnConfig {
    pub input_capacity: usize,
    pub event_capacity: usize,
    pub max_concurrent_operations: usize,
    /// After cancellation, allow cooperative cleanup before dropping the future.
    /// Dropping a future leaves the external outcome indeterminate.
    pub cancellation_grace: Duration,
}

impl Default for TurnConfig {
    fn default() -> Self {
        Self {
            input_capacity: 64,
            event_capacity: 256,
            max_concurrent_operations: 32,
            cancellation_grace: Duration::from_secs(5),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Live model output, committed history, and lifecycle notifications.
pub enum TurnEvent {
    SessionReady {
        connection_id: String,
    },
    Session {
        connection_id: String,
        event: ModelSessionEvent,
    },
    Suspended {
        suspension: Suspension,
    },
    ItemCommitted {
        item: TurnItem,
    },
    Model {
        inference_id: String,
        event: ModelStreamEvent,
    },
    Hook {
        event: FunctionHookEvent,
    },
    Idle,
    Finished {
        outcome: TurnOutcome,
    },
}

/// Cloneable ingress. Journal commands acknowledge persistence; transport input
/// acknowledges local queueing. Neither means a tool or model has finished.
#[derive(Clone)]
pub struct TurnHandle {
    sender: mpsc::Sender<Command>,
    abort: TurnAbortHandle,
    model_input: ModelIngress,
    bidirectional: bool,
}

type ModelIngress = Arc<Mutex<Option<mpsc::Sender<ModelSessionCommand>>>>;

struct InitialInput {
    text: Option<String>,
    metadata: Value,
}

impl TurnHandle {
    /// Requests provider-side cancellation without cancelling tool operations.
    /// Completion is reported by a later ResponseFinished event.
    pub async fn cancel_response(&self, response_id: impl Into<String>) -> Result<()> {
        self.send_model_command(ModelSessionCommand::CancelResponse {
            response_id: response_id.into(),
        })
        .await
    }

    async fn send_model_command(&self, command: ModelSessionCommand) -> Result<()> {
        let sender = self
            .model_input
            .lock()
            .map_err(|_| closed())?
            .clone()
            .ok_or_else(|| {
                AgentError::Model("model connection is not ready or is suspended".into())
            })?;
        let permit = sender.reserve().await.map_err(|_| closed())?;
        let current = self.model_input.lock().map_err(|_| closed())?;
        if !current
            .as_ref()
            .is_some_and(|active| active.same_channel(&sender))
        {
            return Err(closed());
        }
        // Seal/reconnect and local admission share this brief critical section.
        // A sender waiting for capacity cannot enqueue into an obsolete session.
        permit.send(command);
        Ok(())
    }
    /// Text uses journal-first ingress in both model modes. Media is queued
    /// directly; audio bytes are never appended to the thread journal.
    /// Success does not mean the provider has acknowledged delivery.
    pub async fn send_input(&self, input: ModelInput) -> Result<()> {
        if let ModelInput::Text { text } = input {
            return self.append_input(text).await.map(|_| ());
        }
        if !self.bidirectional {
            return Err(AgentError::Model(
                "request-based models accept text input only".into(),
            ));
        }
        self.send_model_command(ModelSessionCommand::Input(input))
            .await
    }

    /// Reopens a suspended model connection from durable history. It does not
    /// replay media or retry tools; the caller explicitly controls this decision.
    pub async fn reconnect_model(&self) -> Result<()> {
        let (reply, result) = oneshot::channel();
        self.send(Command::Reconnect { reply }).await?;
        result.await.map_err(|_| closed())?
    }
    /// Persists text, then updates request-based inference or queues it into the
    /// live bidirectional session without cancelling the provider response.
    /// Bidirectional input requires a ready connection. A delivery error after
    /// persistence does not roll back the journal; recovery uses durable history.
    pub async fn append_input(&self, text: impl Into<String>) -> Result<String> {
        let (reply, result) = oneshot::channel();
        self.send(Command::Input {
            text: text.into(),
            reply,
        })
        .await?;
        result.await.map_err(|_| closed())?
    }

    pub async fn cancel_operation(&self, operation_id: impl Into<String>) -> Result<()> {
        let (reply, result) = oneshot::channel();
        self.send(Command::Cancel {
            operation_id: operation_id.into(),
            reply,
        })
        .await?;
        result.await.map_err(|_| closed())?
    }

    /// Optional host-reported progress; it does not trigger inference.
    pub async fn report_progress(
        &self,
        operation_id: impl Into<String>,
        progress: Value,
    ) -> Result<()> {
        let (reply, result) = oneshot::channel();
        self.send(Command::Progress {
            operation_id: operation_id.into(),
            progress,
            reply,
        })
        .await?;
        result.await.map_err(|_| closed())?
    }

    /// Resume a tool suspension or explicitly reconcile an indeterminate operation.
    /// Approval may repeat a side effect; the host must reconcile it first.
    pub async fn resolve_operation(
        &self,
        operation_id: impl Into<String>,
        resolution: SuspensionResolution,
    ) -> Result<()> {
        let (reply, result) = oneshot::channel();
        self.send(Command::Resolve {
            operation_id: operation_id.into(),
            resolution,
            reply,
        })
        .await?;
        result.await.map_err(|_| closed())?
    }

    pub fn abort(&self) {
        self.abort.abort();
    }

    async fn send(&self, command: Command) -> Result<()> {
        self.sender.send(command).await.map_err(|_| closed())
    }
}

/// Live execution owner, distinct from the durable `canary_agent_kernel::Turn`.
/// Dropping it aborts the interaction. Cloned ingress handles do not own it.
pub struct RunningTurn {
    pub thread_id: String,
    pub turn_id: String,
    handle: TurnHandle,
    events: broadcast::Receiver<TurnEvent>,
    task: Option<JoinHandle<Result<TurnOutcome>>>,
}

impl RunningTurn {
    pub fn handle(&self) -> TurnHandle {
        self.handle.clone()
    }

    /// A slow subscriber gets `Lagged`; durable items can be reloaded from storage.
    pub async fn next_event(
        &mut self,
    ) -> std::result::Result<TurnEvent, broadcast::error::RecvError> {
        self.events.recv().await
    }

    pub fn subscribe(&self) -> broadcast::Receiver<TurnEvent> {
        self.events.resubscribe()
    }

    pub async fn wait(mut self) -> Result<TurnOutcome> {
        let mut task = self.task.take().expect("interaction task");
        // Keep ownership while awaiting so cancellation of wait also aborts.
        let result = (&mut task)
            .await
            .map_err(|error| AgentError::TurnExecution(error.to_string()))?;
        result
    }
}

impl Drop for RunningTurn {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

enum Command {
    Reconnect {
        reply: oneshot::Sender<Result<()>>,
    },
    Input {
        text: String,
        reply: oneshot::Sender<Result<String>>,
    },
    Cancel {
        operation_id: String,
        reply: oneshot::Sender<Result<()>>,
    },
    Progress {
        operation_id: String,
        progress: Value,
        reply: oneshot::Sender<Result<()>>,
    },
    Resolve {
        operation_id: String,
        resolution: SuspensionResolution,
        reply: oneshot::Sender<Result<()>>,
    },
}

fn closed() -> AgentError {
    AgentError::TurnExecution("interaction is closed".into())
}

impl Agent {
    /// Opens a bidirectional interaction before any audio or text arrives.
    pub async fn open_stream_turn(
        &self,
        thread_id: &str,
        metadata: Value,
        config: TurnConfig,
    ) -> Result<RunningTurn> {
        if self.model_client.interaction() != ModelInteraction::Bidirectional {
            return Err(AgentError::Model(
                "model does not support streaming input".into(),
            ));
        }
        self.start_turn(
            thread_id,
            Some(InitialInput {
                text: None,
                metadata,
            }),
            config,
        )
        .await
    }
    /// Opens an interaction and starts inference. Only an accepted end_turn call
    /// successfully completes it; a text-only response leaves it open for input.
    /// Metadata initializes a new thread and does not replace existing metadata.
    pub async fn open_turn(
        &self,
        thread_id: &str,
        user_text: impl Into<String>,
        metadata: Value,
        config: TurnConfig,
    ) -> Result<RunningTurn> {
        self.start_turn(
            thread_id,
            Some(InitialInput {
                text: Some(user_text.into()),
                metadata,
            }),
            config,
        )
        .await
    }

    /// Reconstruct an open interaction. Only idempotent started operations retry
    /// automatically; other unfinished operations require explicit reconciliation.
    pub async fn recover_turn(&self, thread_id: &str, config: TurnConfig) -> Result<RunningTurn> {
        self.start_turn(thread_id, None, config).await
    }

    async fn start_turn(
        &self,
        thread_id: &str,
        input: Option<InitialInput>,
        config: TurnConfig,
    ) -> Result<RunningTurn> {
        if config.input_capacity == 0
            || config.event_capacity == 0
            || config.max_concurrent_operations == 0
        {
            return Err(AgentError::TurnExecution(
                "channel capacities and operation capacity must be positive".into(),
            ));
        }
        if self
            .function_registry
            .specs()
            .iter()
            .any(|spec| matches!(spec.name.as_str(), "end_turn" | "cancel_operation"))
        {
            return Err(AgentError::TurnExecution(
                "end_turn and cancel_operation are reserved runtime commands".into(),
            ));
        }
        let guard = self.register_active_turn(thread_id)?;
        let lease = self.session_coordinator.acquire(thread_id).await?;
        let mut thread = match self.store.load(thread_id).await {
            Ok(thread) => thread,
            Err(AgentError::ThreadNotFound(_)) if input.is_some() => Thread::new(thread_id),
            Err(error) => return Err(error),
        };
        let turn_id;
        let recovering = input.is_none();
        if let Some(InitialInput { text, metadata }) = input {
            if thread
                .turns
                .iter()
                .any(|turn| matches!(turn.status, TurnStatus::Running | TurnStatus::Suspended))
                || !ThreadProjection::from_thread(&thread)
                    .active_operations
                    .is_empty()
            {
                return Err(AgentError::TurnAlreadyActive(thread_id.into()));
            }
            if thread.turns.is_empty() {
                thread.metadata = metadata;
            }
            let mut turn = Turn::new();
            turn_id = turn.id.clone();
            turn.push_item(TurnItem::new(
                TurnItemSource::Runtime,
                TurnItemKind::TurnOpened,
            ));
            if let Some(text) = text {
                turn.push_item(TurnItem::new(
                    TurnItemSource::User,
                    TurnItemKind::UserInput {
                        text,
                        response_to: None,
                    },
                ));
            }
            thread.turns.push(turn);
        } else {
            let turn = thread
                .turns
                .last_mut()
                .ok_or_else(|| AgentError::TurnNotFound(thread_id.into()))?;
            if turn.status == TurnStatus::Completed
                || !turn
                    .items
                    .iter()
                    .any(|item| matches!(item.kind, TurnItemKind::TurnOpened))
            {
                return Err(AgentError::TurnExecution("no recoverable turn".into()));
            }
            turn_id = turn.id.clone();
            turn.set_status(TurnStatus::Running);
        }
        thread = self.commit_thread(thread, lease.fence()).await?;
        let (sender, receiver) = mpsc::channel(config.input_capacity);
        let (events, event_receiver) = broadcast::channel(config.event_capacity);
        let model_input = Arc::new(Mutex::new(None));
        let handle = TurnHandle {
            sender,
            abort: guard.handle.clone(),
            model_input: model_input.clone(),
            bidirectional: self.model_client.interaction() == ModelInteraction::Bidirectional,
        };
        let runner = runner::Runner::new(
            self.clone(),
            thread,
            turn_id.clone(),
            config,
            receiver,
            events,
            guard,
            lease,
            model_input,
        );
        let task = tokio::spawn(runner.run(recovering));
        Ok(RunningTurn {
            thread_id: thread_id.into(),
            turn_id,
            handle,
            events: event_receiver,
            task: Some(task),
        })
    }
}
