use lite_agent_runtime::{AgentFunction, FunctionContext, FunctionExecution, FunctionSpec, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct WebSearchConfig {
    pub endpoint: String,
    pub max_results: usize,
    pub timeout: Duration,
}

impl Default for WebSearchConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://api.duckduckgo.com/".to_string(),
            max_results: 8,
            timeout: Duration::from_secs(15),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DuckDuckGoSearch {
    config: WebSearchConfig,
    client: reqwest::Client,
}

impl DuckDuckGoSearch {
    pub fn new(config: WebSearchConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .user_agent("lite-agent-desktop/0.1")
            .build()
            .unwrap_or_default();
        Self { config, client }
    }
}

#[derive(Debug, Deserialize)]
struct DdgResponse {
    #[serde(rename = "Heading")]
    heading: String,
    #[serde(rename = "AbstractText")]
    abstract_text: String,
    #[serde(rename = "AbstractURL")]
    abstract_url: String,
    #[serde(rename = "RelatedTopics", default)]
    related_topics: Vec<RelatedTopic>,
}
#[derive(Debug, Deserialize)]
struct RelatedTopic {
    #[serde(rename = "Text", default)]
    text: String,
    #[serde(rename = "FirstURL", default)]
    url: String,
}

impl AgentFunction for DuckDuckGoSearch {
    fn spec(&self) -> FunctionSpec {
        FunctionSpec {
            name: "web_search".into(),
            description: "使用 DuckDuckGo 搜索公开网页；需要最新信息时使用。".into(),
            parameters: json!({"type":"object","required":["query"],"properties":{"query":{"type":"string"}},"additionalProperties":false}),
        }
    }

    fn call<'a>(
        &'a self,
        args: Value,
        _context: FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FunctionExecution>> + Send + 'a>> {
        Box::pin(async move {
            let query = args
                .get("query")
                .and_then(Value::as_str)
                .filter(|q| !q.trim().is_empty())
                .ok_or_else(
                    || lite_agent_runtime::AgentError::InvalidFunctionArguments {
                        name: "web_search".into(),
                        message: "query 必须是非空字符串".into(),
                    },
                )?;
            let response: DdgResponse = self
                .client
                .get(&self.config.endpoint)
                .query(&[
                    ("q", query),
                    ("format", "json"),
                    ("no_html", "1"),
                    ("skip_disambig", "1"),
                ])
                .send()
                .await
                .map_err(|e| lite_agent_runtime::AgentError::Http(e.to_string()))?
                .error_for_status()
                .map_err(|e| lite_agent_runtime::AgentError::Http(e.to_string()))?
                .json()
                .await
                .map_err(|e| lite_agent_runtime::AgentError::Http(e.to_string()))?;
            let mut results = Vec::new();
            if !response.abstract_text.is_empty() {
                results.push(json!({"title": response.heading, "url": response.abstract_url, "snippet": response.abstract_text}));
            }
            results.extend(
                response
                    .related_topics
                    .into_iter()
                    .filter(|x| !x.text.is_empty())
                    .take(self.config.max_results)
                    .map(|x| json!({"title": x.text, "url": x.url})),
            );
            Ok(FunctionExecution::Completed {
                output: json!({"provider":"DuckDuckGo","query":query,"results":results,"note":"结果来自 DuckDuckGo Instant Answer API，可能不覆盖所有网页。"}),
            })
        })
    }
}
