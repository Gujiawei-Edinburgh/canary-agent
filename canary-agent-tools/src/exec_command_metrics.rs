use crate::exec_command::{ExecCommandMetric, ExecCommandMetricsRecorder};
use prometheus::{HistogramVec, IntCounterVec, Registry};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct PromExecCommandRecorder {
    registry: Arc<Registry>,
    commands: IntCounterVec,
    durations: HistogramVec,
    output_bytes: HistogramVec,
    output_truncated: IntCounterVec,
}

impl PromExecCommandRecorder {
    pub fn new() -> prometheus::Result<Self> {
        Self::with_registry_and_labels(Arc::new(Registry::new()), None)
    }

    pub fn with_registry_and_labels(
        registry: Arc<Registry>,
        labels: Option<HashMap<String, String>>,
    ) -> prometheus::Result<Self> {
        let labels = labels.unwrap_or_default();
        let commands = IntCounterVec::new(
            prometheus::Opts::new("canary_agent_exec_commands_total", "Exec command outcomes")
                .const_labels(labels.clone()),
            &["outcome"],
        )?;
        let durations = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "canary_agent_exec_command_duration_seconds",
                "Exec command duration",
            )
            .const_labels(labels.clone()),
            &["outcome"],
        )?;
        let output_bytes = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "canary_agent_exec_command_output_bytes",
                "Exec command output bytes",
            )
            .const_labels(labels.clone()),
            &["stream"],
        )?;
        let output_truncated = IntCounterVec::new(
            prometheus::Opts::new(
                "canary_agent_exec_command_output_truncated_total",
                "Exec command output truncations",
            )
            .const_labels(labels),
            &["stream"],
        )?;
        registry.register(Box::new(commands.clone()))?;
        registry.register(Box::new(durations.clone()))?;
        registry.register(Box::new(output_bytes.clone()))?;
        registry.register(Box::new(output_truncated.clone()))?;
        Ok(Self {
            registry,
            commands,
            durations,
            output_bytes,
            output_truncated,
        })
    }

    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }
}

impl ExecCommandMetricsRecorder for PromExecCommandRecorder {
    fn record(&self, metric: ExecCommandMetric) {
        match metric {
            ExecCommandMetric::Finished {
                outcome,
                duration,
                stdout_bytes,
                stderr_bytes,
                stdout_truncated,
                stderr_truncated,
            } => {
                self.commands.with_label_values(&[outcome.as_str()]).inc();
                self.durations
                    .with_label_values(&[outcome.as_str()])
                    .observe(duration.as_secs_f64());
                self.output_bytes
                    .with_label_values(&["stdout"])
                    .observe(stdout_bytes as f64);
                self.output_bytes
                    .with_label_values(&["stderr"])
                    .observe(stderr_bytes as f64);
                if stdout_truncated {
                    self.output_truncated.with_label_values(&["stdout"]).inc();
                }
                if stderr_truncated {
                    self.output_truncated.with_label_values(&["stderr"]).inc();
                }
            }
        }
    }
}
