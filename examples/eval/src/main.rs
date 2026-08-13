use lite_agent_eval::{
    ActivationPolicy, AgentActionEvent, AgentActionStatus, ConstraintDelta, ConstraintId,
    ConstraintOperation, EnvironmentController, EnvironmentControllerInput, EnvironmentDecision,
    EnvironmentDecisionSink, EnvironmentDecisionTool, EnvironmentEventKind, EnvironmentFuture,
    EvalError, EvalReport, EvalReportFuture, EvalRunner, EvalRunnerComponents, GraphEnvironment,
    MetricResult, NodeId, ObservationCause, ObservationContent, ObservationRealizer,
    ObservationRealizerInput, Referee, RefereeInput, RuntimeAgentPolicy, TaskCase, TaskNode,
    TaskTransition, TransitionId, TransitionKind, VisibilityChange,
};
use lite_agent_openai::{ChatCompletionsClient, ModelConfig};
use lite_agent_runtime::{
    Agent, AgentConfig, FunctionRegistry, LocalSessionCoordinator, TurnOutcome,
};
use lite_agent_store_json::JsonFileThreadStore;
use lite_agent_tools::{register_web_search_tools, ExaWebSearchConfig};
use serde_json::{json, Value};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[tokio::main]
async fn main() -> lite_agent_eval::Result<()> {
    let state_dir = PathBuf::from(
        env::var("LITE_AGENT_EVAL_STATE_DIR").unwrap_or_else(|_| ".lite-agent-eval".into()),
    );
    let model = required_env("LITE_AGENT_MODEL")?;
    let api_key = required_env("LITE_AGENT_API_KEY")?;
    let base_url =
        env::var("LITE_AGENT_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let reasoning_effort = env::var("LITE_AGENT_REASONING_EFFORT")
        .unwrap_or_else(|_| ModelConfig::default_reasoning_effort());
    let run_id = env::var("LITE_AGENT_EVAL_RUN_ID").unwrap_or_else(|_| default_run_id());

    let store = Arc::new(JsonFileThreadStore::open(&state_dir)?);
    let model_client = Arc::new(ChatCompletionsClient::new(ModelConfig {
        base_url,
        api_key,
        model,
        reasoning_effort,
    }));

    let mut tested_registry = FunctionRegistry::new();
    let exa_config = ExaWebSearchConfig::new(required_env("EXA_API_KEY")?);
    let exa_config = match env::var("EXA_BASE_URL") {
        Ok(base_url) => exa_config.with_base_url(base_url),
        Err(_) => exa_config,
    };
    register_web_search_tools(&mut tested_registry, exa_config);
    let tested_agent = Arc::new(Agent::new(
        role_config(
            "You are the policy being evaluated in a multi-turn research task. Track revisions across turns, use web_search for current facts, prefer requested source types, and cite source URLs. Treat each new user message as an update to the same task rather than an unrelated request.",
        ),
        store.clone(),
        model_client.clone(),
        tested_registry,
        Arc::new(LocalSessionCoordinator::default()),
    ));

    let decision_sink = Arc::new(EnvironmentDecisionSink::default());
    let mut simulated_user_registry = FunctionRegistry::new();
    simulated_user_registry.register(EnvironmentDecisionTool::new(decision_sink.clone()));
    let simulated_user_agent = Arc::new(Agent::new(
        role_config(
            "You are the simulated user controlling a multi-turn evaluation environment. Inspect the current node, its result and evidence obligations, active constraints, outgoing transitions, and the tested policy's latest action. Call environment_decision exactly once. Transition only when the current node's obligations are satisfied; otherwise retry with a specific reason that helps the policy correct its answer. Failed, suspended, aborted, unsupported, or uncited answers require retry. Never skip nodes or return a decision as assistant text.",
        ),
        store.clone(),
        model_client.clone(),
        simulated_user_registry,
        Arc::new(LocalSessionCoordinator::default()),
    ));

    let referee_agent = Arc::new(Agent::new(
        role_config(
            "You are an evaluation referee. Inspect the complete multi-turn trajectory and final environment snapshot. Assess factual correctness, whether revisions were followed, official-source quality, comparison quality, final synthesis, and unnecessary retries. Give a concise qualitative assessment.",
        ),
        store,
        model_client,
        FunctionRegistry::new(),
        Arc::new(LocalSessionCoordinator::default()),
    ));

    let graph = example_case().compile()?;
    let environment = GraphEnvironment::new(
        graph,
        LlmEnvironmentController::new(
            simulated_user_agent,
            decision_sink,
            format!("eval-{run_id}-simulated-user"),
        ),
        StagedObservationRealizer,
    )?;
    let components = EvalRunnerComponents::new(
        RuntimeAgentPolicy::new(tested_agent, format!("eval-{run_id}-tested-policy")),
        LlmReferee::new(referee_agent, format!("eval-{run_id}-referee")),
    );
    let mut runner = EvalRunner::new(environment, components);
    let report = runner.run().await?;
    let snapshot = runner.environment().snapshot()?;
    let event_count = runner.environment().trajectory().len();

    println!("evaluation status: {:?}", snapshot.status);
    println!("overall score: {:?}", report.overall_score);
    println!("{}", serde_json::to_string_pretty(&report)?);
    println!("trajectory events: {event_count}");
    Ok(())
}

fn required_env(name: &str) -> lite_agent_eval::Result<String> {
    env::var(name).map_err(|_| {
        EvalError::Agent(lite_agent_runtime::AgentError::Model(format!(
            "missing {name}"
        )))
    })
}

fn role_config(system_prompt: &str) -> AgentConfig {
    AgentConfig {
        system_prompt: system_prompt.to_string(),
        ..AgentConfig::default()
    }
}

fn default_run_id() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().to_string())
        .unwrap_or_else(|_| "local".to_string())
}

fn example_case() -> TaskCase {
    TaskCase {
        id: "multi-stage-rust-release-research".to_string(),
        version: "1".to_string(),
        start: NodeId::from("discover_release"),
        nodes: vec![
            TaskNode {
                id: NodeId::from("discover_release"),
                constraints: vec![
                    ConstraintDelta {
                        operation: ConstraintOperation::Add {
                            id: ConstraintId::from("topic"),
                            value: json!("the latest stable Rust release"),
                            activation: ActivationPolicy::ExplicitDisclosure,
                        },
                        provenance: Some("initial request".to_string()),
                    },
                    ConstraintDelta {
                        operation: ConstraintOperation::Add {
                            id: ConstraintId::from("source_policy"),
                            value: json!("use a credible current web source and cite its URL"),
                            activation: ActivationPolicy::ExplicitDisclosure,
                        },
                        provenance: Some("initial request".to_string()),
                    },
                ],
                result_obligation: Some(lite_agent_eval::Obligation {
                    id: "identify_release".to_string(),
                    payload: json!({"requires": ["release version", "supporting source URL"]}),
                }),
                evidence_obligation: Some(lite_agent_eval::Obligation {
                    id: "current_source".to_string(),
                    payload: json!({"minimum_urls": 1}),
                }),
                terminal: false,
            },
            TaskNode {
                id: NodeId::from("official_details"),
                constraints: vec![
                    ConstraintDelta {
                        operation: ConstraintOperation::Replace {
                            target: ConstraintId::from("source_policy"),
                            id: ConstraintId::from("official_source_policy"),
                            value: json!("use only official rust-lang.org sources"),
                        },
                        provenance: Some("user revision".to_string()),
                    },
                    ConstraintDelta {
                        operation: ConstraintOperation::Add {
                            id: ConstraintId::from("release_date_requirement"),
                            value: json!("include the exact release date"),
                            activation: ActivationPolicy::ExplicitDisclosure,
                        },
                        provenance: Some("user revision".to_string()),
                    },
                ],
                result_obligation: Some(lite_agent_eval::Obligation {
                    id: "verify_official_details".to_string(),
                    payload: json!({
                        "requires": ["release version", "exact release date"],
                        "allowed_domains": ["rust-lang.org"]
                    }),
                }),
                evidence_obligation: Some(lite_agent_eval::Obligation {
                    id: "official_release_source".to_string(),
                    payload: json!({"minimum_official_urls": 1}),
                }),
                terminal: false,
            },
            TaskNode {
                id: NodeId::from("compare_previous"),
                constraints: vec![ConstraintDelta {
                    operation: ConstraintOperation::Add {
                        id: ConstraintId::from("comparison_requirement"),
                        value: json!(
                            "compare with the immediately previous stable Rust release and explain one concrete user-visible change"
                        ),
                        activation: ActivationPolicy::ExplicitDisclosure,
                    },
                    provenance: Some("follow-up request".to_string()),
                }],
                result_obligation: Some(lite_agent_eval::Obligation {
                    id: "compare_releases".to_string(),
                    payload: json!({
                        "requires": [
                            "previous release version",
                            "previous release date",
                            "one concrete change"
                        ]
                    }),
                }),
                evidence_obligation: Some(lite_agent_eval::Obligation {
                    id: "official_comparison_sources".to_string(),
                    payload: json!({"allowed_domains": ["rust-lang.org"]}),
                }),
                terminal: false,
            },
            TaskNode {
                id: NodeId::from("final_synthesis"),
                constraints: vec![ConstraintDelta {
                    operation: ConstraintOperation::Add {
                        id: ConstraintId::from("final_format"),
                        value: json!({
                            "maximum_words": 180,
                            "required_sections": ["latest release", "comparison", "sources"]
                        }),
                        activation: ActivationPolicy::ExplicitDisclosure,
                    },
                    provenance: Some("final formatting request".to_string()),
                }],
                result_obligation: Some(lite_agent_eval::Obligation {
                    id: "synthesize_result".to_string(),
                    payload: json!({
                        "maximum_words": 180,
                        "must_reconcile_prior_answers": true
                    }),
                }),
                evidence_obligation: Some(lite_agent_eval::Obligation {
                    id: "final_official_sources".to_string(),
                    payload: json!({"minimum_official_urls": 2}),
                }),
                terminal: false,
            },
            TaskNode {
                id: NodeId::from("finished"),
                constraints: vec![ConstraintDelta {
                    operation: ConstraintOperation::Add {
                        id: ConstraintId::from("evaluation_complete"),
                        value: json!(true),
                        activation: ActivationPolicy::AlreadyAuthorized,
                    },
                    provenance: Some("terminal transition".to_string()),
                }],
                result_obligation: None,
                evidence_obligation: None,
                terminal: true,
            },
        ],
        transitions: vec![
            TaskTransition {
                id: TransitionId::from("require_official_details"),
                from: NodeId::from("discover_release"),
                to: NodeId::from("official_details"),
                kind: TransitionKind::Revision,
            },
            TaskTransition {
                id: TransitionId::from("compare_previous_release"),
                from: NodeId::from("official_details"),
                to: NodeId::from("compare_previous"),
                kind: TransitionKind::Progress,
            },
            TaskTransition {
                id: TransitionId::from("request_final_synthesis"),
                from: NodeId::from("compare_previous"),
                to: NodeId::from("final_synthesis"),
                kind: TransitionKind::Progress,
            },
            TaskTransition {
                id: TransitionId::from("finish"),
                from: NodeId::from("final_synthesis"),
                to: NodeId::from("finished"),
                kind: TransitionKind::Progress,
            },
        ],
    }
}

struct StagedObservationRealizer;

impl ObservationRealizer for StagedObservationRealizer {
    fn realize<'a>(
        &'a self,
        input: ObservationRealizerInput,
    ) -> EnvironmentFuture<'a, ObservationContent> {
        Box::pin(async move {
            let (user_text, visibility) = match input.cause {
                ObservationCause::Reset => (
                    "Find the latest stable Rust release. Use web_search, tell me the version, and cite a credible current source URL.".to_string(),
                    vec![
                        disclose("topic"),
                        disclose("source_policy"),
                    ],
                ),
                ObservationCause::Retry { reason } => (
                    format!(
                        "That does not fully satisfy my current request: {reason}. Please correct the same step before we continue."
                    ),
                    Vec::new(),
                ),
                ObservationCause::Transition { transition }
                    if transition.0 == "require_official_details" => (
                        "I want to revise the source requirement: use only official rust-lang.org sources, and include the exact release date.".to_string(),
                        vec![disclose("release_date_requirement")],
                    ),
                ObservationCause::Transition { transition }
                    if transition.0 == "compare_previous_release" => (
                        "Now compare it with the immediately previous stable Rust release. Include that release's date and explain one concrete user-visible change, still using official Rust sources.".to_string(),
                        vec![disclose("comparison_requirement")],
                    ),
                ObservationCause::Transition { transition }
                    if transition.0 == "request_final_synthesis" => (
                        "Give me a final self-contained synthesis in at most 180 words with sections for the latest release, the comparison, and official source URLs. Reconcile any inconsistency in your earlier answers.".to_string(),
                        vec![disclose("final_format")],
                    ),
                ObservationCause::Transition { transition } => {
                    return Err(EvalError::Role(format!(
                        "no observation is defined for transition {}",
                        transition.0
                    )));
                }
            };
            Ok(ObservationContent {
                user_text,
                visibility,
                metadata: json!({
                    "case": "multi-stage-rust-release-research",
                    "node": input.state.current_node,
                }),
            })
        })
    }
}

fn disclose(constraint: &str) -> VisibilityChange {
    VisibilityChange::Disclose {
        constraint: ConstraintId::from(constraint),
    }
}

struct LlmEnvironmentController {
    agent: Arc<Agent>,
    decision_sink: Arc<EnvironmentDecisionSink>,
    thread_id: String,
}

impl LlmEnvironmentController {
    fn new(
        agent: Arc<Agent>,
        decision_sink: Arc<EnvironmentDecisionSink>,
        thread_id: String,
    ) -> Self {
        Self {
            agent,
            decision_sink,
            thread_id,
        }
    }
}

impl EnvironmentController for LlmEnvironmentController {
    fn decide<'a>(
        &'a self,
        input: EnvironmentControllerInput,
    ) -> EnvironmentFuture<'a, EnvironmentDecision> {
        Box::pin(async move {
            if self.decision_sink.take().is_some() {
                return Err(EvalError::Role(
                    "simulated-user decision sink contained a stale decision".to_string(),
                ));
            }
            let prompt = format!(
                "Decide how the environment should react to this tested-policy action. Call environment_decision exactly once. A transition decision must name an outgoing transition from the current node. Use an empty evidence array when no concrete reference can be extracted.\n\n{}",
                serde_json::to_string_pretty(&input)?
            );
            require_completed_turn(
                self.agent
                    .run_turn(
                        &self.thread_id,
                        prompt,
                        json!({"role": "environment_controller"}),
                        |_| {},
                    )
                    .await?,
                "simulated user",
            )?;
            self.decision_sink.take().ok_or_else(|| {
                EvalError::Role(
                    "simulated user completed without calling environment_decision".to_string(),
                )
            })
        })
    }
}

struct LlmReferee {
    agent: Arc<Agent>,
    thread_id: String,
}

impl LlmReferee {
    fn new(agent: Arc<Agent>, thread_id: String) -> Self {
        Self { agent, thread_id }
    }
}

impl Referee for LlmReferee {
    fn evaluate<'a>(&'a self, input: RefereeInput) -> EvalReportFuture<'a, EvalReport> {
        Box::pin(async move {
            let prompt = format!(
                "Assess this completed evaluation. Explain whether the tested policy identified the latest release, honored the official-source revision, compared the previous release accurately, and produced the requested final synthesis. Note any unnecessary retries or unsupported claims. Do not emit JSON; your assessment will be attached to deterministic metrics.\n\n{}",
                serde_json::to_string_pretty(&input)?
            );
            let assessment = require_completed_turn(
                self.agent
                    .run_turn(&self.thread_id, prompt, json!({"role": "referee"}), |_| {})
                    .await?,
                "referee",
            )?;

            let actions = input
                .trajectory
                .iter()
                .filter_map(|event| match &event.kind {
                    EnvironmentEventKind::AgentActionRecorded { action } => Some(action),
                    _ => None,
                });
            let actions = actions.collect::<Vec<_>>();
            let answer_present = actions.iter().any(|action| {
                action.status == AgentActionStatus::Completed
                    && !action.assistant_text.trim().is_empty()
            });
            let web_search_used = actions.iter().any(|action| {
                action.events.iter().any(|event| {
                    matches!(event, AgentActionEvent::FunctionCompleted { name, .. } if name == "web_search")
                })
            });
            let source_cited = actions.iter().any(|action| {
                action.assistant_text.contains("https://")
                    || action.assistant_text.contains("http://")
            });
            let official_source_cited = actions
                .iter()
                .any(|action| action.assistant_text.contains("rust-lang.org"));
            let transitions_applied = input
                .trajectory
                .iter()
                .filter(|event| {
                    matches!(event.kind, EnvironmentEventKind::TransitionApplied { .. })
                })
                .count();
            let completed_all_stages = transitions_applied == 4 && actions.len() >= 4;
            let metrics = vec![
                boolean_metric("answer_present", answer_present),
                boolean_metric("web_search_used", web_search_used),
                boolean_metric("source_cited", source_cited),
                boolean_metric("official_source_cited", official_source_cited),
                boolean_metric("completed_all_stages", completed_all_stages),
            ];
            let overall_score =
                Some(metrics.iter().map(|metric| metric.score).sum::<f64>() / metrics.len() as f64);
            Ok(EvalReport {
                metrics,
                overall_score,
                details: json!({"referee_assessment": assessment}),
            })
        })
    }
}

fn require_completed_turn(outcome: TurnOutcome, role: &str) -> lite_agent_eval::Result<String> {
    match outcome {
        TurnOutcome::AssistantMessage { text } => Ok(text),
        TurnOutcome::Suspended { suspension } => Err(EvalError::Role(format!(
            "{role} suspended: {:?}: {}",
            suspension.kind, suspension.payload
        ))),
        TurnOutcome::Failed { error } => {
            Err(EvalError::Role(format!("{role} turn failed: {error}")))
        }
        TurnOutcome::Aborted { reason } => {
            Err(EvalError::Role(format!("{role} turn aborted: {reason}")))
        }
    }
}

fn boolean_metric(name: &str, passed: bool) -> MetricResult {
    MetricResult {
        name: name.to_string(),
        score: if passed { 1.0 } else { 0.0 },
        passed: Some(passed),
        details: Value::Null,
    }
}
