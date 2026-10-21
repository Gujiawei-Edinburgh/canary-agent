//! Open interactions with interruptible inference and independently tracked tools.
mod runner;
mod worker;

use super::*;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::JoinHandle;

#[derive(Debug, Clone)]
pub struct FullDuplexConfig {
    pub input_capacity: usize,
    pub event_capacity: usize,
    pub max_concurrent_operations: usize,
    /// After cancellation, allow cooperative cleanup before dropping the future.
    /// Dropping a future leaves the external outcome indeterminate.
    pub cancellation_grace: Duration,
}

impl Default for FullDuplexConfig {
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
pub enum FullDuplexEvent {
    ItemCommitted {
        item: TurnItem,
    },
    Model {
        inference_id: String,
        event: ModelStreamEvent,
    },
    Hook {
        event: TurnStreamEvent,
    },
    Idle,
    Finished {
        outcome: TurnOutcome,
    },
}

/// Cloneable ingress. Acknowledgments mean that the event has been persisted.
#[derive(Clone)]
pub struct FullDuplexHandle {
    sender: mpsc::Sender<Command>,
    abort: TurnAbortHandle,
}

impl FullDuplexHandle {
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

/// Dropping this owner aborts the interaction. Cloned ingress handles do not own it.
pub struct FullDuplexTurn {
    pub thread_id: String,
    pub turn_id: String,
    handle: FullDuplexHandle,
    events: broadcast::Receiver<FullDuplexEvent>,
    task: Option<JoinHandle<Result<TurnOutcome>>>,
}

impl FullDuplexTurn {
    pub fn handle(&self) -> FullDuplexHandle {
        self.handle.clone()
    }

    /// A slow subscriber gets `Lagged`; durable items can be reloaded from storage.
    pub async fn next_event(
        &mut self,
    ) -> std::result::Result<FullDuplexEvent, broadcast::error::RecvError> {
        self.events.recv().await
    }

    pub fn subscribe(&self) -> broadcast::Receiver<FullDuplexEvent> {
        self.events.resubscribe()
    }

    pub async fn wait(mut self) -> Result<TurnOutcome> {
        let mut task = self.task.take().expect("interaction task");
        // Keep ownership while awaiting so cancellation of wait also aborts.
        let result = (&mut task)
            .await
            .map_err(|error| AgentError::FullDuplex(error.to_string()))?;
        result
    }
}

impl Drop for FullDuplexTurn {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

enum Command {
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
    AgentError::FullDuplex("interaction is closed".into())
}

impl Agent {
    pub async fn open_full_duplex_turn(
        &self,
        thread_id: &str,
        user_text: impl Into<String>,
        metadata: Value,
        config: FullDuplexConfig,
    ) -> Result<FullDuplexTurn> {
        self.start_full_duplex(thread_id, Some((user_text.into(), metadata)), config)
            .await
    }

    /// Reconstruct an open interaction. Only idempotent started operations retry
    /// automatically; other unfinished operations require explicit reconciliation.
    pub async fn recover_full_duplex_turn(
        &self,
        thread_id: &str,
        config: FullDuplexConfig,
    ) -> Result<FullDuplexTurn> {
        self.start_full_duplex(thread_id, None, config).await
    }

    async fn start_full_duplex(
        &self,
        thread_id: &str,
        input: Option<(String, Value)>,
        config: FullDuplexConfig,
    ) -> Result<FullDuplexTurn> {
        if config.input_capacity == 0
            || config.event_capacity == 0
            || config.max_concurrent_operations == 0
        {
            return Err(AgentError::FullDuplex(
                "channel capacities and operation capacity must be positive".into(),
            ));
        }
        if self
            .function_registry
            .specs()
            .iter()
            .any(|spec| matches!(spec.name.as_str(), "end_turn" | "cancel_operation"))
        {
            return Err(AgentError::FullDuplex(
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
        if let Some((text, metadata)) = input {
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
                TurnItemKind::FullDuplexOpened,
            ));
            turn.push_item(TurnItem::new(
                TurnItemSource::User,
                TurnItemKind::UserInput {
                    text,
                    response_to: None,
                },
            ));
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
                    .any(|item| matches!(item.kind, TurnItemKind::FullDuplexOpened))
            {
                return Err(AgentError::FullDuplex(
                    "no recoverable full-duplex turn".into(),
                ));
            }
            turn_id = turn.id.clone();
            turn.set_status(TurnStatus::Running);
        }
        thread = self.commit_thread(thread, lease.fence()).await?;
        let (sender, receiver) = mpsc::channel(config.input_capacity);
        let (events, event_receiver) = broadcast::channel(config.event_capacity);
        let handle = FullDuplexHandle {
            sender,
            abort: self
                .active_turns
                .lock()
                .map_err(|_| closed())?
                .get(thread_id)
                .ok_or_else(closed)?
                .handle
                .clone(),
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
        );
        let task = tokio::spawn(runner.run(recovering));
        Ok(FullDuplexTurn {
            thread_id: thread_id.into(),
            turn_id,
            handle,
            events: event_receiver,
            task: Some(task),
        })
    }
}

#[cfg(test)]
mod tests;
