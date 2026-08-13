use lite_agent_openai::{ChatCompletionsClient, ModelConfig};
use lite_agent_runtime::{Agent, AgentConfig, FunctionRegistry, LocalSessionCoordinator};
use lite_agent_store_json::JsonFileThreadStore;
use lite_agent_tools::{register_web_search_tools, ExaWebSearchConfig};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main]
async fn main() -> lite_agent_eval::Result<()> {
    let state_dir = PathBuf::from(
        env::var("LITE_AGENT_EVAL_STATE_DIR").unwrap_or_else(|_| ".lite-agent-eval".into()),
    );
    let model = env::var("LITE_AGENT_MODEL")
        .map_err(|_| lite_agent_runtime::AgentError::Model("missing LITE_AGENT_MODEL".into()))?;
    let api_key = env::var("LITE_AGENT_API_KEY")
        .map_err(|_| lite_agent_runtime::AgentError::Model("missing LITE_AGENT_API_KEY".into()))?;
    let base_url =
        env::var("LITE_AGENT_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let reasoning_effort = env::var("LITE_AGENT_REASONING_EFFORT")
        .unwrap_or_else(|_| ModelConfig::default_reasoning_effort());

    let store = Arc::new(JsonFileThreadStore::open(&state_dir)?);
    let model_client = Arc::new(ChatCompletionsClient::new(ModelConfig {
        base_url,
        api_key,
        model,
        reasoning_effort,
    }));
    let exa_api_key = env::var("EXA_API_KEY")
        .map_err(|_| lite_agent_runtime::AgentError::Model("missing EXA_API_KEY".to_string()))?;
    let mut tested_registry = FunctionRegistry::new();
    let exa_config = ExaWebSearchConfig::new(exa_api_key);
    let exa_config = match env::var("EXA_BASE_URL") {
        Ok(base_url) => exa_config.with_base_url(base_url),
        Err(_) => exa_config,
    };
    register_web_search_tools(&mut tested_registry, exa_config);
    let _tested_agent = Arc::new(Agent::new(
        AgentConfig::default(),
        store.clone(),
        model_client.clone(),
        tested_registry,
        Arc::new(LocalSessionCoordinator::default()),
    ));
    Ok(())
}
