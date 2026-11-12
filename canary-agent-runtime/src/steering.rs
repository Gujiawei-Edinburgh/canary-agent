//! Cooperative, iteration-boundary input admission. Queueing never cancels work.
use crate::{AgentError, Result};
use canary_agent_kernel::{TurnItem, TurnItemKind, TurnItemSource};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};
use tokio::sync::oneshot;

/// Local queue acceptance, not durable recording or model consumption.
/// Pending input is in memory; it is not recoverable after a process crash.
pub struct SteeringReceipt {
    pub input_id: String,
    recorded: oneshot::Receiver<std::result::Result<(), String>>,
}

impl SteeringReceipt {
    /// Wait until the input has committed to the current turn's journal.
    /// Fails if the turn exits, aborts, suspends, or persistence fails before
    /// the next input boundary. Dropping this receipt does not withdraw input.
    pub async fn wait_recorded(self) -> Result<()> {
        let result = self
            .recorded
            .await
            .unwrap_or_else(|_| Err("turn owner was dropped before recording input".into()));
        result.map_err(|reason| AgentError::SteeringInputUnconfirmed {
            input_id: self.input_id,
            reason,
        })
    }
}

pub(crate) struct PendingInput {
    pub item: TurnItem,
    recorded: oneshot::Sender<std::result::Result<(), String>>,
}

impl PendingInput {
    pub fn recorded(self) {
        let _ = self.recorded.send(Ok(()));
    }
    pub fn rejected(self, reason: &str) {
        let _ = self.recorded.send(Err(reason.into()));
    }
}

struct State {
    accepting: bool,
    pending: VecDeque<PendingInput>,
}

#[derive(Clone)]
pub(crate) struct InputQueue {
    state: Arc<Mutex<State>>,
    capacity: usize,
}

impl InputQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                accepting: true,
                pending: VecDeque::new(),
            })),
            capacity,
        }
    }

    pub fn enqueue(&self, text: String) -> Result<SteeringReceipt> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| AgentError::SessionCoordinator("steering queue poisoned".into()))?;
        if !state.accepting {
            return Err(AgentError::SteeringClosed);
        }
        if state.pending.len() >= self.capacity {
            return Err(AgentError::SteeringQueueFull {
                capacity: self.capacity,
            });
        }
        let item = TurnItem::new(
            TurnItemSource::User,
            TurnItemKind::UserInput {
                text,
                response_to: None,
            },
        );
        let (recorded, receiver) = oneshot::channel();
        let receipt = SteeringReceipt {
            input_id: item.id.clone(),
            recorded: receiver,
        };
        state.pending.push_back(PendingInput { item, recorded });
        Ok(receipt)
    }

    pub fn drain(&self) -> Result<Vec<PendingInput>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| AgentError::SessionCoordinator("steering queue poisoned".into()))?;
        Ok(state.pending.drain(..).collect())
    }

    /// Completion and admission share one lock. If input wins, the loop must
    /// run another iteration; if completion wins, further input is rejected.
    pub fn seal_if_empty(&self) -> Result<bool> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| AgentError::SessionCoordinator("steering queue poisoned".into()))?;
        if !state.pending.is_empty() {
            return Ok(false);
        }
        state.accepting = false;
        Ok(true)
    }

    pub fn close(&self, reason: &str) {
        if let Ok(mut state) = self.state.lock() {
            state.accepting = false;
            for pending in state.pending.drain(..) {
                pending.rejected(reason);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fifo_capacity_recording_and_closed_admission_are_distinct() {
        let queue = InputQueue::new(2);
        let first = queue.enqueue("first".into()).unwrap();
        let second = queue.enqueue("second".into()).unwrap();
        assert!(matches!(
            queue.enqueue("third".into()),
            Err(AgentError::SteeringQueueFull { capacity: 2 })
        ));
        assert!(!queue.seal_if_empty().unwrap());
        let pending = queue.drain().unwrap();
        assert_eq!(pending[0].item.id, first.input_id);
        assert_eq!(pending[1].item.id, second.input_id);
        for input in pending {
            input.recorded();
        }
        first.wait_recorded().await.unwrap();
        second.wait_recorded().await.unwrap();
        assert!(queue.seal_if_empty().unwrap());
        assert!(matches!(
            queue.enqueue("late".into()),
            Err(AgentError::SteeringClosed)
        ));
    }

    #[tokio::test]
    async fn closing_rejects_unrecorded_receipts() {
        let queue = InputQueue::new(1);
        let receipt = queue.enqueue("pending".into()).unwrap();
        queue.close("turn suspended before the next boundary");
        assert!(matches!(
            receipt.wait_recorded().await,
            Err(AgentError::SteeringInputUnconfirmed { .. })
        ));
    }

    #[test]
    fn completion_racing_admission_never_loses_accepted_input() {
        for _ in 0..32 {
            let queue = InputQueue::new(1);
            let barrier = Arc::new(std::sync::Barrier::new(2));
            std::thread::scope(|scope| {
                let producer = scope.spawn(|| {
                    barrier.wait();
                    queue.enqueue("racing input".into())
                });
                barrier.wait();
                let sealed = queue.seal_if_empty().unwrap();
                let receipt = producer.join().unwrap();
                if sealed {
                    assert!(matches!(receipt, Err(AgentError::SteeringClosed)));
                } else {
                    assert_eq!(queue.drain().unwrap()[0].item.id, receipt.unwrap().input_id);
                }
            });
        }
    }
}
