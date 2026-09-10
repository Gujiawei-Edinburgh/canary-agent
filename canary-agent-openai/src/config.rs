use std::time::Duration;

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_backoff: Duration::from_millis(250),
            max_backoff: Duration::from_secs(4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub reasoning_effort: String,
}

impl ModelConfig {
    pub fn default_reasoning_effort() -> String {
        "medium".to_string()
    }
}
