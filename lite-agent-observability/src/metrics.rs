use lite_agent_runtime::{MetricsRecorder, RuntimeMetric};
use prometheus::{Encoder, HistogramVec, IntCounterVec, Registry, TextEncoder};
use std::sync::Arc;

#[derive(Clone)]
pub struct PromRecorder {
    registry: Arc<Registry>,
    turns: IntCounterVec,
    turns_per_thread: IntCounterVec,
    turn_latency: HistogramVec,
    function_calls_per_turn: HistogramVec,
    ttft: HistogramVec,
    model_requests: HistogramVec,
    function_calls: HistogramVec,
    token_usage: IntCounterVec,
}

impl PromRecorder {
    pub fn new() -> prometheus::Result<Self> {
        Self::with_registry(Arc::new(Registry::new()))
    }

    pub fn with_registry(registry: Arc<Registry>) -> prometheus::Result<Self> {
        let turns = IntCounterVec::new(
            prometheus::Opts::new("lite_agent_turns_total", "Agent turn outcomes"),
            &["status"],
        )?;
        let turns_per_thread = IntCounterVec::new(
            prometheus::Opts::new("lite_agent_turns_per_thread_total", "Agent turns by thread"),
            &["thread", "status"],
        )?;
        let turn_latency = HistogramVec::new(
            prometheus::HistogramOpts::new("lite_agent_turn_latency_seconds", "Agent turn latency"),
            &["status"],
        )?;
        let function_calls_per_turn = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "lite_agent_function_calls_per_turn",
                "Function calls per turn",
            ),
            &[],
        )?;
        let ttft = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "lite_agent_time_to_first_token_seconds",
                "Time to first streamed token",
            ),
            &[],
        )?;
        let model_requests = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "lite_agent_model_request_duration_seconds",
                "Model request duration",
            ),
            &["status"],
        )?;
        let function_calls = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "lite_agent_function_call_duration_seconds",
                "Function call duration",
            ),
            &["function", "status"],
        )?;
        let token_usage = IntCounterVec::new(
            prometheus::Opts::new("lite_agent_tokens_total", "Model token usage"),
            &["kind"],
        )?;
        registry.register(Box::new(turns.clone()))?;
        registry.register(Box::new(turns_per_thread.clone()))?;
        registry.register(Box::new(turn_latency.clone()))?;
        registry.register(Box::new(function_calls_per_turn.clone()))?;
        registry.register(Box::new(ttft.clone()))?;
        registry.register(Box::new(model_requests.clone()))?;
        registry.register(Box::new(function_calls.clone()))?;
        registry.register(Box::new(token_usage.clone()))?;
        Ok(Self {
            registry,
            turns,
            turns_per_thread,
            turn_latency,
            function_calls_per_turn,
            ttft,
            model_requests,
            function_calls,
            token_usage,
        })
    }

    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }

    pub fn gather(&self) -> Vec<prometheus::proto::MetricFamily> {
        self.registry.gather()
    }

    pub fn encode(&self) -> prometheus::Result<String> {
        let mut buffer = Vec::new();
        TextEncoder::new()
            .encode(&self.gather(), &mut buffer)
            .map_err(|error| prometheus::Error::Msg(error.to_string()))?;
        String::from_utf8(buffer).map_err(|error| prometheus::Error::Msg(error.to_string()))
    }
}

impl MetricsRecorder for PromRecorder {
    fn record(&self, metric: RuntimeMetric) {
        match metric {
            RuntimeMetric::TurnFinished {
                thread_id,
                status,
                duration,
                function_calls,
            } => {
                self.turns.with_label_values(&[status.as_str()]).inc();
                self.turns_per_thread
                    .with_label_values(&[&thread_id, status.as_str()])
                    .inc();
                self.turn_latency
                    .with_label_values(&[status.as_str()])
                    .observe(duration.as_secs_f64());
                self.function_calls_per_turn
                    .with_label_values(&[])
                    .observe(function_calls as f64);
            }
            RuntimeMetric::ModelRequestFinished { status, duration } => {
                self.model_requests
                    .with_label_values(&[status.as_str()])
                    .observe(duration.as_secs_f64());
            }
            RuntimeMetric::FunctionCallFinished {
                name,
                status,
                duration,
            } => {
                self.function_calls
                    .with_label_values(&[&name, status.as_str()])
                    .observe(duration.as_secs_f64());
            }
            RuntimeMetric::TimeToFirstToken { duration } => {
                self.ttft
                    .with_label_values(&[])
                    .observe(duration.as_secs_f64());
            }
            RuntimeMetric::TokenUsage {
                input_tokens,
                cached_input_tokens,
                output_tokens,
                total_tokens,
            } => {
                self.token_usage
                    .with_label_values(&["input"])
                    .inc_by(input_tokens);
                self.token_usage
                    .with_label_values(&["cached_input"])
                    .inc_by(cached_input_tokens);
                self.token_usage
                    .with_label_values(&["output"])
                    .inc_by(output_tokens);
                self.token_usage
                    .with_label_values(&["total"])
                    .inc_by(total_tokens);
            }
        }
    }
}
