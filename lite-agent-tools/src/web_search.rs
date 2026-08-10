use crate::web_search_metrics::{
    NoopWebSearchMetricsRecorder, WebSearchMetric, WebSearchMetricsRecorder, WebSearchOutcome,
};
use lite_agent_runtime::{
    AgentError, AgentFunction, DiscardResolver, FunctionContext, FunctionExecution, FunctionLimits,
    FunctionOutputResolver, FunctionRecoveryPolicy, FunctionRegistry, FunctionSpec, Result,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.exa.ai";
const DEFAULT_MAX_RESULTS: usize = 5;
const MAX_RESULTS: usize = 20;
const MAX_OUTPUT_BYTES: usize = 512 * 1024;

/// Configuration for the Exa-backed `web_search` function.
#[derive(Debug, Clone)]
pub struct ExaWebSearchConfig {
    pub api_key: String,
    pub base_url: String,
    pub default_max_results: usize,
    pub timeout: Duration,
}

impl ExaWebSearchConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            default_max_results: DEFAULT_MAX_RESULTS,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_default_max_results(mut self, max_results: usize) -> Self {
        self.default_max_results = max_results.clamp(1, MAX_RESULTS);
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[derive(Clone)]
pub struct ExaWebSearchTool {
    client: Client,
    config: ExaWebSearchConfig,
    metrics_recorder: Arc<dyn WebSearchMetricsRecorder>,
}

impl ExaWebSearchTool {
    pub fn new(config: ExaWebSearchConfig) -> Self {
        Self {
            client: Client::new(),
            config,
            metrics_recorder: Arc::new(NoopWebSearchMetricsRecorder),
        }
    }

    pub fn with_metrics_recorder<R>(mut self, recorder: R) -> Self
    where
        R: WebSearchMetricsRecorder + 'static,
    {
        self.metrics_recorder = Arc::new(recorder);
        self
    }

    async fn search(&self, request: SearchRequest) -> Result<SearchOutput> {
        let max_results = request
            .max_results
            .unwrap_or(self.config.default_max_results);
        let body = ExaSearchRequest {
            query: request.query,
            num_results: max_results,
            include_domains: request.include_domains,
            exclude_domains: request.exclude_domains,
            contents: ExaContents { highlights: true },
        };
        let url = format!("{}/search", self.config.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(url)
            .header("x-api-key", &self.config.api_key)
            .header("accept", "application/json")
            .json(&body)
            .timeout(self.config.timeout)
            .send()
            .await
            .map_err(|error| AgentError::Http(format!("Exa web search request failed: {error}")))?;
        let status = response.status();
        let raw = response.text().await.map_err(|error| {
            AgentError::Http(format!("failed to read Exa web search response: {error}"))
        })?;
        if !status.is_success() {
            return Err(AgentError::Http(format!(
                "Exa web search returned HTTP {status}: {raw}"
            )));
        }
        let response: ExaSearchResponse = serde_json::from_str(&raw).map_err(|error| {
            AgentError::Http(format!("invalid Exa web search response: {error}"))
        })?;
        Ok(SearchOutput {
            query: body.query,
            results: response.results,
        })
    }
}

impl AgentFunction for ExaWebSearchTool {
    fn spec(&self) -> FunctionSpec {
        FunctionSpec {
            name: "web_search".to_string(),
            description: "Search the live web and return ranked sources with relevant highlights. Use this for current or externally sourced information; cite the returned URLs in the answer.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query."
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_RESULTS,
                        "description": "Maximum number of sources to return."
                    },
                    "include_domains": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional domain allowlist, such as [\"docs.rs\"]."
                    },
                    "exclude_domains": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional domains to exclude."
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        }
    }

    fn limits(&self) -> FunctionLimits {
        FunctionLimits {
            time_budget: self.config.timeout,
            max_output_bytes: MAX_OUTPUT_BYTES,
        }
    }

    fn recovery_policy(&self) -> FunctionRecoveryPolicy {
        FunctionRecoveryPolicy::Idempotent
    }

    fn output_resolver(&self) -> &dyn FunctionOutputResolver {
        &DiscardResolver
    }

    fn call<'a>(
        &'a self,
        args: Value,
        _context: FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FunctionExecution>> + Send + 'a>> {
        Box::pin(async move {
            let started = std::time::Instant::now();
            let result = async {
                let request: SearchRequest = serde_json::from_value(args).map_err(|error| {
                    AgentError::InvalidFunctionArguments {
                        name: "web_search".to_string(),
                        message: error.to_string(),
                    }
                })?;
                if request.query.trim().is_empty() {
                    return Err(AgentError::InvalidFunctionArguments {
                        name: "web_search".to_string(),
                        message: "query must not be empty".to_string(),
                    });
                }
                if let Some(max_results) = request.max_results {
                    if !(1..=MAX_RESULTS).contains(&max_results) {
                        return Err(AgentError::InvalidFunctionArguments {
                            name: "web_search".to_string(),
                            message: format!("max_results must be between 1 and {MAX_RESULTS}"),
                        });
                    }
                }
                self.search(request).await
            }
            .await;
            let (result_count, response_bytes) = result
                .as_ref()
                .ok()
                .map(|output| {
                    let result_count = output.results.len();
                    let response_bytes = serde_json::to_vec(output).map_or(0, |bytes| bytes.len());
                    (result_count, response_bytes)
                })
                .unwrap_or_default();
            self.metrics_recorder.record(WebSearchMetric::Finished {
                outcome: web_search_outcome(&result),
                duration: started.elapsed(),
                result_count,
                response_bytes,
            });
            let output = result?;
            Ok(FunctionExecution::Completed {
                output: serde_json::to_value(output)?,
            })
        })
    }
}

fn web_search_outcome(result: &Result<SearchOutput>) -> WebSearchOutcome {
    match result {
        Ok(_) => WebSearchOutcome::Succeeded,
        Err(AgentError::InvalidFunctionArguments { .. }) => WebSearchOutcome::InvalidArguments,
        Err(AgentError::Http(_)) => WebSearchOutcome::HttpError,
        Err(_) => WebSearchOutcome::Failed,
    }
}

pub fn register_web_search_tools(registry: &mut FunctionRegistry, config: ExaWebSearchConfig) {
    registry.register(ExaWebSearchTool::new(config));
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchRequest {
    query: String,
    max_results: Option<usize>,
    include_domains: Option<Vec<String>>,
    exclude_domains: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct ExaSearchRequest {
    query: String,
    #[serde(rename = "numResults")]
    num_results: usize,
    #[serde(rename = "includeDomains", skip_serializing_if = "Option::is_none")]
    include_domains: Option<Vec<String>>,
    #[serde(rename = "excludeDomains", skip_serializing_if = "Option::is_none")]
    exclude_domains: Option<Vec<String>>,
    contents: ExaContents,
}

#[derive(Debug, Serialize)]
struct ExaContents {
    highlights: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExaSearchResponse {
    #[serde(default)]
    results: Vec<ExaSearchResult>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExaSearchResult {
    title: Option<String>,
    url: String,
    #[serde(rename = "publishedDate", skip_serializing_if = "Option::is_none")]
    published_date: Option<String>,
    author: Option<String>,
    #[serde(default)]
    highlights: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SearchOutput {
    query: String,
    results: Vec<ExaSearchResult>,
}

#[cfg(test)]
mod tests {
    use super::{ExaSearchRequest, ExaSearchResponse, ExaWebSearchConfig, ExaWebSearchTool};
    use lite_agent_runtime::AgentFunction;
    use serde_json::json;

    #[test]
    fn exposes_web_search_contract() {
        let tool = ExaWebSearchTool::new(ExaWebSearchConfig::new("test-key"));
        let spec = tool.spec();
        assert_eq!(spec.name, "web_search");
        assert_eq!(spec.parameters["required"], json!(["query"]));
        assert_eq!(spec.parameters["additionalProperties"], false);
    }

    #[test]
    fn serializes_exa_search_request_with_highlights() {
        let request = ExaSearchRequest {
            query: "rust async runtime".to_string(),
            num_results: 3,
            include_domains: Some(vec!["rust-lang.org".to_string()]),
            exclude_domains: None,
            contents: super::ExaContents { highlights: true },
        };
        let value = serde_json::to_value(request).expect("request json");
        assert_eq!(value["numResults"], 3);
        assert_eq!(value["includeDomains"], json!(["rust-lang.org"]));
        assert_eq!(value["contents"]["highlights"], true);
    }

    #[test]
    fn decodes_search_results_and_ignores_unneeded_provider_fields() {
        let response: ExaSearchResponse = serde_json::from_value(json!({
            "results": [{
                "title": "Rust",
                "url": "https://www.rust-lang.org/",
                "publishedDate": "2026-01-01T00:00:00Z",
                "author": "Rust team",
                "highlights": ["A relevant excerpt"],
                "favicon": "https://www.rust-lang.org/favicon.ico"
            }]
        }))
        .expect("response json");
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].url, "https://www.rust-lang.org/");
        assert_eq!(response.results[0].highlights[0], "A relevant excerpt");
    }
}
