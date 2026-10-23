//! Provider-neutral contracts for a persistent bidirectional model connection.
use crate::{ModelDescriptor, ModelRequest, Result};
pub use canary_agent_kernel::ModelResponseStatus;
use canary_agent_kernel::{ModelFunctionCall, Thread, TokenUsage, TurnItem};
use std::future::Future;
use std::pin::Pin;

pub type ModelSessionFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelInteraction {
    RequestResponse,
    Bidirectional,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFormat {
    pub encoding: AudioEncoding,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioEncoding {
    PcmS16Le,
    PcmF32Le,
    Opus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioChunk {
    pub format: AudioFormat,
    pub data: Vec<u8>,
}

/// Transport input, distinct from a journal append. Audio chunks belong to the
/// ongoing connection; provider-reported item IDs identify recognized utterances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelInput {
    Text { text: String },
    Audio { chunk: AudioChunk },
    EndAudio,
}

pub struct ModelSessionConfig {
    pub connection_id: String,
    pub turn_id: String,
    pub request: ModelRequest,
    /// Durable facts for adapter-specific restoration, including unresolved calls.
    /// Reconnecting must never re-execute a tool or replay old microphone chunks.
    pub history: Thread,
    pub model: ModelDescriptor,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelSessionCommand {
    Input(ModelInput),
    /// Already committed tool lifecycle facts. The adapter maps these to native
    /// results/observations; it must not invent successful execution receipts.
    Observe {
        items: Vec<TurnItem>,
    },
    CancelResponse {
        response_id: String,
    },
    Close,
}

/// Adapters assemble function arguments before ItemCompleted. IDs are stable
/// within a connection; one item is finalized once. Transcripts describe input
/// already consumed and must never be submitted again as fresh model input.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelSessionEvent {
    InputStarted {
        item_id: String,
    },
    InputEnded {
        item_id: String,
    },
    InputCommitted {
        item_id: String,
    },
    TranscriptFinalized {
        item_id: String,
        text: String,
    },
    ResponseStarted {
        response_id: String,
    },
    TextDelta {
        response_id: String,
        item_id: String,
        text: String,
    },
    AudioDelta {
        response_id: String,
        item_id: String,
        chunk: AudioChunk,
    },
    ItemCompleted {
        response_id: String,
        item_id: String,
        text: Option<String>,
        function_calls: Vec<ModelFunctionCall>,
    },
    ResponseFinished {
        response_id: String,
        status: ModelResponseStatus,
    },
    /// A usage delta, not a cumulative snapshot. Duplicate event IDs are ignored.
    Usage {
        event_id: String,
        response_id: String,
        usage: TokenUsage,
    },
}

pub trait ModelSessionSender: Send {
    fn send(&mut self, command: ModelSessionCommand) -> ModelSessionFuture<'_, ()>;
}

pub trait ModelSessionReceiver: Send {
    /// None means the connection closed, not that the turn completed.
    /// Cancelling this future must not lose an event.
    fn receive(&mut self) -> ModelSessionFuture<'_, Option<ModelSessionEvent>>;
}

/// Independently owned halves allow input to flow while output is received.
/// Dropping them must release the underlying connection resources.
pub struct ModelSession {
    pub sender: Box<dyn ModelSessionSender>,
    pub receiver: Box<dyn ModelSessionReceiver>,
}
