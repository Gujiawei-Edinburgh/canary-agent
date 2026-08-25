use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricStatus {
    Completed,
    Suspended,
    Failed,
    Aborted,
}

impl MetricStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Suspended => "suspended",
            Self::Failed => "failed",
            Self::Aborted => "aborted",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionCallOutcome {
    Completed,
    Suspended,
    Failed,
    Aborted,
}

impl FunctionCallOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Suspended => "suspended",
            Self::Failed => "failed",
            Self::Aborted => "aborted",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionCallSkipReason {
    MaxCallsPerTurn,
    PreviousFunctionSuspended,
    NonIdempotentRecovery,
    TurnAborted,
}

impl FunctionCallSkipReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MaxCallsPerTurn => "max_calls_per_turn",
            Self::PreviousFunctionSuspended => "previous_function_suspended",
            Self::NonIdempotentRecovery => "non_idempotent_recovery",
            Self::TurnAborted => "turn_aborted",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeMetric {
    TurnFinished {
        status: MetricStatus,
        duration: Duration,
        function_calls: usize,
    },
    ModelRequestFinished {
        status: MetricStatus,
        duration: Duration,
    },
    FunctionCallFinished {
        name: String,
        outcome: FunctionCallOutcome,
        duration: Duration,
    },
    FunctionCallSkipped {
        name: String,
        reason: FunctionCallSkipReason,
    },
    TimeToFirstToken {
        duration: Duration,
    },
    TokenUsage {
        input_tokens: u64,
        cached_input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
    },
}

pub trait MetricsRecorder: Send + Sync {
    fn record(&self, metric: RuntimeMetric);
}

#[derive(Debug, Default)]
pub struct NoopMetricsRecorder;

impl MetricsRecorder for NoopMetricsRecorder {
    fn record(&self, _metric: RuntimeMetric) {}
}
