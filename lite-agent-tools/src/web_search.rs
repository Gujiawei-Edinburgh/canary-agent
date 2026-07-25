use lite_agent_runtime::{AgentFunction, FunctionContext, FunctionExecution, FunctionSpec, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

const TOOL_NAME: &str = "web_search";

#[derive(Debug, Clone)]
pub struct BaiduSearchConfig {
    pub endpoint: String,
    pub api_key: String,
    pub max_results: usize,
    pub timeout: Duration,
}

impl Default for BaiduSearchConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://qianfan.baidubce.com/v2/ai_search/web_search".to_string(),
            api_key: String::new(),
            max_results: 8,
            timeout: Duration::from_secs(20),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BaiduWebSearch {
    config: BaiduSearchConfig,
    client: reqwest::Client,
}

impl BaiduWebSearch {
    pub fn new(config: BaiduSearchConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .user_agent("lite-agent-desktop/0.1")
            .build()
            .unwrap_or_default();
        Self { config, client }
    }
}

#[derive(Debug, Deserialize)]
struct BaiduResponse {
    #[serde(default)]
    code: Option<Value>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    references: Vec<BaiduReference>,
}

#[derive(Debug, Deserialize)]
struct BaiduReference {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    snippet: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    date: String,
    #[serde(default)]
    website: String,
    #[serde(default)]
    web_anchor: String,
    #[serde(rename = "type", default)]
    resource_type: String,
}

fn api_error(response: &BaiduResponse) -> Option<String> {
    let code = response.code.as_ref()?;
    let successful = match code {
        Value::Null => true,
        Value::Number(number) => number.as_i64() == Some(0),
        Value::String(value) => {
            value.is_empty() || value == "0" || value.eq_ignore_ascii_case("success")
        }
        _ => false,
    };
    if successful {
        None
    } else {
        Some(format!(
            "千帆百度搜索返回错误 {code}: {}",
            response.message.as_deref().unwrap_or("未知错误")
        ))
    }
}

fn normalized_results(response: BaiduResponse, max_results: usize) -> Vec<Value> {
    response
        .references
        .into_iter()
        .filter(|reference| reference.resource_type.is_empty() || reference.resource_type == "web")
        .filter(|reference| !reference.url.trim().is_empty())
        .take(max_results)
        .map(|reference| {
            let title = [
                reference.title,
                reference.web_anchor,
                reference.website.clone(),
            ]
            .into_iter()
            .find(|value| !value.trim().is_empty())
            .unwrap_or_else(|| reference.url.clone());
            let snippet = if reference.snippet.trim().is_empty() {
                reference.content
            } else {
                reference.snippet
            };
            json!({
                "title": title,
                "url": reference.url,
                "snippet": snippet,
                "date": reference.date,
                "source": reference.website,
            })
        })
        .collect()
}

impl AgentFunction for BaiduWebSearch {
    fn spec(&self) -> FunctionSpec {
        FunctionSpec {
            name: TOOL_NAME.into(),
            description: "使用千帆百度搜索查询公开网页和实时信息，返回网页标题、链接、摘要和日期。"
                .into(),
            parameters: json!({
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "简洁的搜索关键词或问题"
                    }
                },
                "additionalProperties": false
            }),
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
                .map(str::trim)
                .filter(|query| !query.is_empty())
                .ok_or_else(
                    || lite_agent_runtime::AgentError::InvalidFunctionArguments {
                        name: TOOL_NAME.into(),
                        message: "query 必须是非空字符串".into(),
                    },
                )?;
            let api_key = self.config.api_key.trim();
            if api_key.is_empty() {
                return Err(lite_agent_runtime::AgentError::Function {
                    name: TOOL_NAME.into(),
                    message: "尚未配置千帆 API Key，请先打开桌面应用设置并填写千帆配置".into(),
                });
            }

            let response = self
                .client
                .post(&self.config.endpoint)
                .bearer_auth(api_key)
                .json(&json!({
                    "messages": [{"role": "user", "content": query}],
                    "search_source": "baidu_search_v2",
                    "resource_type_filter": [{"type": "web", "top_k": self.config.max_results}],
                    "sort": {"priority": "auto"}
                }))
                .send()
                .await
                .map_err(|error| lite_agent_runtime::AgentError::Http(error.to_string()))?;
            let status = response.status();
            let body = response
                .text()
                .await
                .map_err(|error| lite_agent_runtime::AgentError::Http(error.to_string()))?;
            if !status.is_success() {
                let detail = body.chars().take(500).collect::<String>();
                return Err(lite_agent_runtime::AgentError::Http(format!(
                    "千帆百度搜索请求失败（HTTP {status}）：{detail}"
                )));
            }
            let response: BaiduResponse = serde_json::from_str(&body).map_err(|error| {
                lite_agent_runtime::AgentError::Http(format!("千帆百度搜索响应解析失败：{error}"))
            })?;
            if let Some(message) = api_error(&response) {
                return Err(lite_agent_runtime::AgentError::Function {
                    name: TOOL_NAME.into(),
                    message,
                });
            }
            let results = normalized_results(response, self.config.max_results);
            Ok(FunctionExecution::Completed {
                output: json!({
                    "provider": "百度千帆",
                    "query": query,
                    "results": results,
                }),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{api_error, normalized_results, BaiduResponse, BaiduSearchConfig, BaiduWebSearch};
    use lite_agent_runtime::AgentFunction;

    #[test]
    fn exposes_baidu_search_tool() {
        let tool = BaiduWebSearch::new(BaiduSearchConfig::default());
        let spec = tool.spec();
        assert_eq!(spec.name, "web_search");
        assert!(spec.description.contains("百度"));
        assert_eq!(spec.parameters["additionalProperties"], false);
    }

    #[test]
    fn normalizes_web_references_and_ignores_other_media() {
        let response: BaiduResponse = serde_json::from_value(serde_json::json!({
            "references": [
                {"title":"结果一","url":"https://example.com/1","snippet":"摘要一","date":"2026-07-25","website":"示例站点","type":"web"},
                {"title":"图片","url":"https://example.com/image","type":"image"},
                {"title":"结果二","url":"https://example.com/2","content":"摘要二","type":"web"}
            ]
        }))
        .expect("valid response");

        let results = normalized_results(response, 8);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["title"], "结果一");
        assert_eq!(results[0]["snippet"], "摘要一");
        assert_eq!(results[1]["snippet"], "摘要二");
    }

    #[test]
    fn surfaces_api_errors_returned_with_successful_http_status() {
        let response: BaiduResponse = serde_json::from_value(serde_json::json!({
            "code": "invalid_api_key",
            "message": "API key is invalid"
        }))
        .expect("valid response");
        assert!(api_error(&response)
            .expect("API error")
            .contains("invalid_api_key"));
    }
}
