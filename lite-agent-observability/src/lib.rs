mod jsonl_collector;
mod logging;
mod metrics;

pub use jsonl_collector::JsonlTraceCollector;
pub use lite_agent_runtime::NoopMetricsRecorder as NoopRecorder;
pub use logging::{init_file_logging, LoggingError, LoggingGuard};
pub use metrics::PromRecorder;

#[cfg(test)]
mod tests {
    use super::JsonlTraceCollector;
    use lite_agent_runtime::{TraceCollector, TraceEvent, TraceEventKind};
    use serde_json::json;
    use std::fs;

    #[test]
    fn prom_recorder_exposes_runtime_metrics() {
        use lite_agent_runtime::{MetricStatus, MetricsRecorder, RuntimeMetric};

        let recorder = super::PromRecorder::new().expect("prometheus recorder");
        recorder.record(RuntimeMetric::TurnFinished {
            thread_id: "thread-1".to_string(),
            status: MetricStatus::Completed,
            duration: std::time::Duration::from_millis(10),
            function_calls: 2,
        });
        recorder.record(RuntimeMetric::TokenUsage {
            input_tokens: 3,
            cached_input_tokens: 1,
            output_tokens: 2,
            total_tokens: 5,
        });

        let output = recorder.encode().expect("metrics encoding");
        assert!(output.contains("lite_agent_turns_total"));
        assert!(output.contains("lite_agent_tokens_total"));
    }

    #[tokio::test]
    async fn writes_one_file_per_thread_and_flushes_records() {
        let temp = tempfile::tempdir().expect("tempdir");
        let collector = JsonlTraceCollector::new(temp.path()).expect("collector");
        collector.record(TraceEvent {
            thread_id: "thread_a".to_string(),
            turn_id: "turn_a".to_string(),
            sequence: 1,
            occurred_at: "1".to_string(),
            kind: TraceEventKind::UserInput {
                text: "hello".to_string(),
                response_to: None,
            },
        });
        collector.record(TraceEvent {
            thread_id: "thread_b".to_string(),
            turn_id: "turn_b".to_string(),
            sequence: 1,
            occurred_at: "2".to_string(),
            kind: TraceEventKind::ToolOutput {
                call_id: "call_b".to_string(),
                name: "echo".to_string(),
                result: lite_agent_kernel::ToolResult::Success {
                    output: json!({"ok": true}),
                },
            },
        });
        collector.flush().await;

        let first =
            fs::read_to_string(temp.path().join("traces/thread_a.jsonl")).expect("thread a trace");
        let second =
            fs::read_to_string(temp.path().join("traces/thread_b.jsonl")).expect("thread b trace");
        assert_eq!(first.lines().count(), 1);
        assert_eq!(second.lines().count(), 1);
        assert!(first.contains("hello"));
        assert!(second.contains("call_b"));
    }
}
