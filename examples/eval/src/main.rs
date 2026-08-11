use lite_agent_eval::{
    ActivationPolicy, AgentInput, AgentObservation, EvalCommandSink, EvalCommandTool, EvalError,
    EvalReport, EvalVm, EvalVmComponents, ProcessorInput, Referee, RefereeInput, RoleFuture,
    RuntimeAgentIo, SimulatedUserCommand, SimulatedUserProcessor, TaskCase, TaskNode,
    TaskTransition, TransitionId, TransitionKind,
};
use lite_agent_openai::{ChatCompletionsClient, ModelConfig};
use lite_agent_runtime::{Agent, AgentConfig, FunctionRegistry, LocalSessionCoordinator};
use lite_agent_store_json::JsonFileThreadStore;
use lite_agent_tools::{register_web_search_tools, ExaWebSearchConfig};
use serde_json::json;
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
    let tested_agent = Arc::new(Agent::new(
        AgentConfig::default(),
        store.clone(),
        model_client.clone(),
        tested_registry,
        Arc::new(LocalSessionCoordinator::default()),
    ));

    let command_sink = Arc::new(EvalCommandSink::default());
    let mut simulated_user_registry = FunctionRegistry::new();
    simulated_user_registry.register(EvalCommandTool::new(command_sink.clone()));
    let simulated_user_agent = Arc::new(Agent::new(
        AgentConfig::default(),
        store.clone(),
        model_client.clone(),
        simulated_user_registry,
        Arc::new(LocalSessionCoordinator::default()),
    ));
    let referee_agent = Arc::new(Agent::new(
        AgentConfig::default(),
        store,
        model_client,
        FunctionRegistry::new(),
        Arc::new(LocalSessionCoordinator::default()),
    ));

    let tested_agent = RuntimeAgentIo::new(tested_agent);
    let simulated_user =
        LlmSimulatedUser::new(RuntimeAgentIo::new(simulated_user_agent), command_sink);
    let referee = LlmReferee::new(RuntimeAgentIo::new(referee_agent));
    let program = example_program().compile()?;
    let components = EvalVmComponents::new(tested_agent, simulated_user, referee);
    let mut vm = EvalVm::new(program, components)?;
    let report = vm.run().await?;

    println!("evaluation status: {:?}", vm.projection().status);
    println!("overall score: {:?}", report.overall_score);
    println!(
        "metrics: {}",
        serde_json::to_string_pretty(&report.metrics)?
    );
    println!("events: {}", vm.events().len());
    Ok(())
}

fn example_program() -> TaskCase {
    TaskCase {
        id: "generic-two-step-agent-task".to_string(),
        version: "1".to_string(),
        start: "start".into(),
        nodes: vec![
            TaskNode {
                id: "start".into(),
                constraints: vec![lite_agent_eval::ConstraintDelta {
                    operation: lite_agent_eval::ConstraintOperation::Add {
                        id: "topic".into(),
                        value: json!("a current fact supported by a web source"),
                        activation: ActivationPolicy::ExplicitDisclosure,
                    },
                    provenance: Some("example".to_string()),
                }],
                result_obligation: None,
                evidence_obligation: None,
                terminal: false,
            },
            TaskNode {
                id: "finished".into(),
                constraints: vec![lite_agent_eval::ConstraintDelta {
                    operation: lite_agent_eval::ConstraintOperation::Add {
                        id: "response_received".into(),
                        value: json!(true),
                        activation: ActivationPolicy::AlreadyAuthorized,
                    },
                    provenance: Some("example".to_string()),
                }],
                result_obligation: None,
                evidence_obligation: None,
                terminal: true,
            },
        ],
        transitions: vec![TaskTransition {
            id: TransitionId::from("finish"),
            from: "start".into(),
            to: "finished".into(),
            kind: TransitionKind::Progress,
            user_message: Some(
                "Use web_search to find one current fact about Rust and briefly cite the source."
                    .to_string(),
            ),
        }],
    }
}

struct LlmSimulatedUser {
    io: RuntimeAgentIo,
    command_sink: Arc<EvalCommandSink>,
}

impl LlmSimulatedUser {
    fn new(io: RuntimeAgentIo, command_sink: Arc<EvalCommandSink>) -> Self {
        Self { io, command_sink }
    }
}

impl SimulatedUserProcessor for LlmSimulatedUser {
    fn decide<'a>(&'a self, input: ProcessorInput) -> RoleFuture<'a, SimulatedUserCommand> {
        Box::pin(async move {
            let prompt = format!(
                "You are the simulated user inside a generic evaluation VM. Decide the next VM action. You have an eval_command tool. Always call that tool exactly once; never return the command as assistant text. When phase is AwaitingUserAction, send a message using the outgoing transition's user_message. After observing a completed agent response, commit that pending transition. If the response is inadequate, retry instead. Evidence entries must be objects with {{\"kind\":\"url\",\"reference\":\"https://...\"}}.\n\nVM input:\n{}",
                serde_json::to_string_pretty(&input)?
            );
            let observation = self
                .io
                .run(AgentInput {
                    thread_id: "eval-simulated-user".to_string(),
                    user_text: prompt,
                })
                .await?;
            let _ = observation;
            self.command_sink.take().ok_or_else(|| {
                EvalError::Agent(lite_agent_runtime::AgentError::Model(
                    "simulated user did not call eval_command".to_string(),
                ))
            })
        })
    }
}

struct LlmReferee {
    io: RuntimeAgentIo,
}

impl LlmReferee {
    fn new(io: RuntimeAgentIo) -> Self {
        Self { io }
    }
}

impl Referee for LlmReferee {
    fn evaluate<'a>(&'a self, input: RefereeInput) -> RoleFuture<'a, EvalReport> {
        Box::pin(async move {
            let prompt = format!(
                "You are the referee for a generic agent evaluation. Inspect the final VM projection and factual events. Return only JSON matching EvalReport: {{\"metrics\":[{{\"name\":\"task_completion\",\"score\":1.0,\"passed\":true,\"details\":{{}}}}],\"overall_score\":1.0,\"details\":{{}}}}. Score task completion based on whether the VM reached its terminal node and the tested agent produced a non-empty response.\n\nInput:\n{}",
                serde_json::to_string_pretty(&input)?
            );
            let observation = self
                .io
                .run(AgentInput {
                    thread_id: "eval-referee".to_string(),
                    user_text: prompt,
                })
                .await?;
            parse_json(&observation, "referee report")
        })
    }
}

fn parse_json<T>(observation: &AgentObservation, role: &str) -> lite_agent_eval::Result<T>
where
    T: serde::de::DeserializeOwned,
{
    let text = observation.assistant_text.trim();
    let text = text
        .strip_prefix("```json")
        .and_then(|value| value.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(text);
    serde_json::from_str(text).map_err(|error| {
        EvalError::Agent(lite_agent_runtime::AgentError::Model(format!(
            "{role} returned invalid JSON: {error}"
        )))
    })
}
