use prometheus::{HistogramVec, IntCounterVec, Registry};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSearchOutcome {
    Succeeded,
    InvalidArguments,
    HttpError,
    Failed,
}

impl WebSearchOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::InvalidArguments => "invalid_arguments",
            Self::HttpError => "http_error",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum WebSearchMetric {
    Finished {
        outcome: WebSearchOutcome,
        duration: Duration,
        result_count: usize,
        response_bytes: usize,
    },
}

pub trait WebSearchMetricsRecorder: Send + Sync {
    fn record(&self, metric: WebSearchMetric);
}

#[derive(Debug, Default)]
pub struct NoopWebSearchMetricsRecorder;

impl WebSearchMetricsRecorder for NoopWebSearchMetricsRecorder {
    fn record(&self, _metric: WebSearchMetric) {}
}

#[derive(Clone)]
pub struct PromWebSearchRecorder {
    registry: Arc<Registry>,
    searches: IntCounterVec,
    durations: HistogramVec,
    result_counts: HistogramVec,
    response_bytes: HistogramVec,
}

impl PromWebSearchRecorder {
    pub fn new() -> prometheus::Result<Self> {
        Self::with_registry_and_labels(Arc::new(Registry::new()), None)
    }

    pub fn with_registry_and_labels(
        registry: Arc<Registry>,
        labels: Option<HashMap<String, String>>,
    ) -> prometheus::Result<Self> {
        let labels = labels.unwrap_or_default();
        let searches = IntCounterVec::new(
            prometheus::Opts::new("canary_agent_web_searches_total", "Web search outcomes")
                .const_labels(labels.clone()),
            &["outcome"],
        )?;
        let durations = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "canary_agent_web_search_duration_seconds",
                "Web search duration",
            )
            .const_labels(labels.clone()),
            &["outcome"],
        )?;
        let result_counts = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "canary_agent_web_search_result_count",
                "Number of results returned by web search",
            )
            .const_labels(labels.clone()),
            &["outcome"],
        )?;
        let response_bytes = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "canary_agent_web_search_response_bytes",
                "Serialized web search response size",
            )
            .const_labels(labels),
            &["outcome"],
        )?;
        registry.register(Box::new(searches.clone()))?;
        registry.register(Box::new(durations.clone()))?;
        registry.register(Box::new(result_counts.clone()))?;
        registry.register(Box::new(response_bytes.clone()))?;
        Ok(Self {
            registry,
            searches,
            durations,
            result_counts,
            response_bytes,
        })
    }

    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }
}

impl WebSearchMetricsRecorder for PromWebSearchRecorder {
    fn record(&self, metric: WebSearchMetric) {
        match metric {
            WebSearchMetric::Finished {
                outcome,
                duration,
                result_count,
                response_bytes,
            } => {
                let outcome = outcome.as_str();
                self.searches.with_label_values(&[outcome]).inc();
                self.durations
                    .with_label_values(&[outcome])
                    .observe(duration.as_secs_f64());
                self.result_counts
                    .with_label_values(&[outcome])
                    .observe(result_count as f64);
                self.response_bytes
                    .with_label_values(&[outcome])
                    .observe(response_bytes as f64);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PromWebSearchRecorder, WebSearchMetric, WebSearchOutcome};
    use crate::WebSearchMetricsRecorder;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn records_web_search_metrics() {
        let recorder = PromWebSearchRecorder::new().expect("recorder");
        recorder.record(WebSearchMetric::Finished {
            outcome: WebSearchOutcome::Succeeded,
            duration: Duration::from_millis(25),
            result_count: 3,
            response_bytes: 1024,
        });
        let metric_families = recorder.registry().gather();
        assert!(metric_families
            .iter()
            .any(|family| family.get_name() == "canary_agent_web_searches_total"));
        assert!(metric_families
            .iter()
            .any(|family| family.get_name() == "canary_agent_web_search_duration_seconds"));
    }

    #[test]
    fn supports_shared_registry() {
        let registry = Arc::new(prometheus::Registry::new());
        let recorder = PromWebSearchRecorder::with_registry_and_labels(registry.clone(), None)
            .expect("recorder");
        assert!(Arc::ptr_eq(&recorder.registry(), &registry));
    }
}
