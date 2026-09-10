use crate::config::{ModelConfig, RetryConfig};
use canary_agent_runtime::{AgentError, Result};
use serde::Serialize;

pub(crate) async fn send_with_retries<T: Serialize + ?Sized>(
    http: &reqwest::Client,
    config: &ModelConfig,
    retry: &RetryConfig,
    url: &str,
    body: &T,
) -> Result<reqwest::Response> {
    for retry_index in 0..=retry.max_retries {
        let response = http
            .post(url)
            .bearer_auth(&config.api_key)
            .json(body)
            .send()
            .await;
        match response {
            Ok(response) if response.status().is_success() => return Ok(response),
            Ok(response) => {
                let status = response.status();
                let raw = response
                    .text()
                    .await
                    .map_err(|error| AgentError::Http(error.to_string()))?;
                if retry_index == retry.max_retries || !is_retryable_status(status) {
                    return Err(AgentError::Http(format!(
                        "HTTP status {status} for streamed model request: {raw}"
                    )));
                }
                wait_before_retry(retry, retry_index).await;
                tracing::warn!(
                    status = %status,
                    retry = retry_index + 1,
                    max_retries = retry.max_retries,
                    "retrying model request"
                );
            }
            Err(error) => {
                let retryable = error.is_connect() || error.is_timeout();
                if retry_index == retry.max_retries || !retryable {
                    return Err(AgentError::Http(error.to_string()));
                }
                wait_before_retry(retry, retry_index).await;
                tracing::warn!(
                    error = %error,
                    retry = retry_index + 1,
                    max_retries = retry.max_retries,
                    "retrying model request"
                );
            }
        }
    }
    unreachable!("retry loop always returns a response or error")
}

async fn wait_before_retry(retry: &RetryConfig, retry_index: usize) {
    let multiplier = 2u32.saturating_pow(retry_index.min(31) as u32);
    let delay = retry
        .initial_backoff
        .checked_mul(multiplier)
        .unwrap_or(retry.max_backoff)
        .min(retry.max_backoff);
    tokio::time::sleep(delay).await;
}
pub(crate) fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 425 | 429 | 500..=599)
}

pub(crate) fn find_sse_frame_end(buffer: &[u8]) -> Option<(usize, usize)> {
    for index in 0..buffer.len() {
        if buffer.get(index..index + 2) == Some(b"\n\n") {
            return Some((index, 2));
        }
        if buffer.get(index..index + 4) == Some(b"\r\n\r\n") {
            return Some((index, 4));
        }
    }
    None
}
