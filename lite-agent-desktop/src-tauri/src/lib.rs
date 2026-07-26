use lite_agent_kernel::events::{new_id, Thread};
use lite_agent_observability::{init_file_logging, JsonlTraceCollector, LoggingGuard};
use lite_agent_openai::{ChatCompletionsClient, ModelConfig};
use lite_agent_runtime::{
    builtin_registry, Agent, AgentConfig, CompactingContextBuilder, FunctionContext,
    LocalSessionCoordinator, Result, RuntimeEvent, ThreadStore, TraceCollector, TurnModelEvent,
    TurnOutcome, TurnStateEvent, TurnStreamEvent,
};
use lite_agent_store_json::JsonFileThreadStore;
use lite_agent_tools::sandbox::{SandboxBackend, SandboxPolicy};
use lite_agent_tools::{
    register_time_tools, register_web_search, AuthorizationDecision, BaiduSearchConfig,
    ExecAuthorizer, ExecCommandConfig, ExecCommandTool,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tauri::{Emitter, Manager, State};

const DEFAULT_MAX_CONTEXT_TOKENS: usize = 256 * 1024;
const DEFAULT_MAX_MODEL_ITERATIONS: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DesktopConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    pub qianfan_api_key: String,
    pub workspace: String,
    pub max_context_tokens: usize,
    pub max_model_iterations: usize,
}

impl Default for DesktopConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            model: String::new(),
            api_key: String::new(),
            qianfan_api_key: String::new(),
            workspace: String::new(),
            max_context_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
            max_model_iterations: DEFAULT_MAX_MODEL_ITERATIONS,
        }
    }
}

struct AppState {
    config: Mutex<DesktopConfig>,
    runtime: Mutex<Option<Runtime>>,
    runtime_error: Mutex<Option<String>>,
    _logging_guard: LoggingGuard,
    diagnostics_dir: PathBuf,
}
struct Runtime {
    agent: Agent,
    store: Arc<JsonFileThreadStore>,
    trace_collector: Arc<JsonlTraceCollector>,
    state_dir: PathBuf,
}

fn error(message: impl ToString) -> tauri::Error {
    tauri::Error::Io(std::io::Error::other(message.to_string()))
}

fn config_path(app: &tauri::AppHandle) -> tauri::Result<PathBuf> {
    Ok(app.path().app_data_dir()?.join("config.json"))
}
fn state_dir(app: &tauri::AppHandle) -> tauri::Result<PathBuf> {
    Ok(app.path().app_data_dir()?.join("state"))
}
fn load_config(app: &tauri::AppHandle) -> tauri::Result<DesktopConfig> {
    let p = config_path(app)?;
    if !p.exists() {
        return Ok(DesktopConfig::default());
    };
    let raw = std::fs::read_to_string(p).map_err(error)?;
    serde_json::from_str(&raw).map_err(error)
}
fn persist_config(app: &tauri::AppHandle, c: &DesktopConfig) -> tauri::Result<()> {
    let p = config_path(app)?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(error)?
    };
    std::fs::write(p, serde_json::to_vec_pretty(c).unwrap()).map_err(error)
}

struct DenyAuthorizer;
impl ExecAuthorizer for DenyAuthorizer {
    fn authorize<'a>(
        &'a self,
        _: &'a lite_agent_tools::ExecRequest,
        _: &'a SandboxPolicy,
        _: &'a str,
        _: &'a FunctionContext,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AuthorizationDecision>> + Send + 'a>,
    > {
        Box::pin(async {
            Ok(AuthorizationDecision::Deny {
                reason: "桌面版 MVP 尚未接入交互式命令审批".into(),
            })
        })
    }
}
struct WorkspaceResolver {
    fallback: PathBuf,
}
impl lite_agent_tools::WorkspaceResolver for WorkspaceResolver {
    fn resolve<'a>(
        &'a self,
        r: lite_agent_tools::WorkspaceResolveRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<lite_agent_tools::ResolvedWorkspace>>
                + Send
                + 'a,
        >,
    > {
        let fallback = self.fallback.clone();
        Box::pin(async move {
            let cwd = r
                .metadata
                .get("workspace")
                .and_then(Value::as_str)
                .map(PathBuf::from)
                .filter(|p| p.is_absolute())
                .unwrap_or(fallback);
            Ok(lite_agent_tools::ResolvedWorkspace {
                policy: SandboxPolicy::workspace_read_write_with_host_network(cwd.clone()),
                cwd,
            })
        })
    }
}

fn backend() -> Arc<dyn SandboxBackend> {
    #[cfg(target_os = "macos")]
    {
        return Arc::new(lite_agent_tools::sandbox::MacOsSeatbeltBackend::new());
    }
    #[cfg(target_os = "linux")]
    {
        return Arc::new(lite_agent_tools::sandbox::LinuxNativeBackend::new());
    }
    #[cfg(windows)]
    {
        return Arc::new(lite_agent_tools::sandbox::WindowsFakeBackend::new());
    }
    #[allow(unreachable_code)]
    Arc::new(lite_agent_tools::sandbox::MacOsSeatbeltBackend::new())
}
fn build_runtime(c: &DesktopConfig, dir: PathBuf) -> tauri::Result<Runtime> {
    if !(8_192..=2_000_000).contains(&c.max_context_tokens) {
        return Err(error("上下文窗口必须在 8,192 到 2,000,000 tokens 之间"));
    }
    if !(1..=65_535).contains(&c.max_model_iterations) {
        return Err(error("Model iteration 上限必须在 1 到 65,535 之间"));
    }
    let workspace = if c.workspace.trim().is_empty() {
        std::env::current_dir().map_err(error)?
    } else {
        PathBuf::from(&c.workspace)
    };
    if !workspace.exists() {
        std::fs::create_dir_all(&workspace)
            .map_err(|e| error(format!("创建 workspace {} 失败: {e}", workspace.display())))?;
    }
    std::fs::create_dir_all(dir.join("threads"))
        .map_err(|e| error(format!("创建会话目录失败: {e}")))?;
    let store = Arc::new(JsonFileThreadStore::open(&dir).map_err(error)?);
    let trace_collector = Arc::new(JsonlTraceCollector::new(&dir).map_err(error)?);
    let model = Arc::new(ChatCompletionsClient::new(ModelConfig {
        base_url: c.base_url.clone(),
        api_key: c.api_key.clone(),
        model: c.model.clone(),
        reasoning_effort: ModelConfig::default_reasoning_effort(),
    }));
    let mut reg = builtin_registry();
    register_time_tools(&mut reg);
    register_web_search(
        &mut reg,
        BaiduSearchConfig {
            api_key: c.qianfan_api_key.clone(),
            ..BaiduSearchConfig::default()
        },
    );
    reg.register(ExecCommandTool::new(
        ExecCommandConfig::new(
            workspace.clone(),
            backend(),
            SandboxPolicy::workspace_read_write_with_host_network(&workspace),
            Arc::new(DenyAuthorizer),
        )
        .with_workspace_resolver(WorkspaceResolver {
            fallback: workspace,
        }),
    ));
    let agent = Agent::new(
        AgentConfig {
            max_model_iterations: c.max_model_iterations,
            system_prompt: "你是 lite-agent，使用中文回答。需要执行命令时使用 exec_command，需要查询公开网页时使用 web_search，需要当前时间时使用 get_current_time。".into(),
        },
        store.clone(),
        model,
        reg,
        Arc::new(LocalSessionCoordinator::default()),
    )
    .with_context_builder(CompactingContextBuilder {
        max_context_tokens: c.max_context_tokens,
        ..CompactingContextBuilder::default()
    })
    .with_shared_trace_collector(trace_collector.clone());
    tracing::info!(
        max_context_tokens = c.max_context_tokens,
        max_model_iterations = c.max_model_iterations,
        state_dir = %dir.display(),
        "desktop runtime initialized"
    );
    Ok(Runtime {
        agent,
        store,
        trace_collector,
        state_dir: dir,
    })
}

#[tauri::command]
fn get_config(state: State<'_, AppState>) -> DesktopConfig {
    state.config.lock().unwrap().clone()
}

#[tauri::command]
fn get_diagnostics_dir(state: State<'_, AppState>) -> String {
    state.diagnostics_dir.display().to_string()
}

#[tauri::command]
fn get_runtime_error(state: State<'_, AppState>) -> Option<String> {
    state.runtime_error.lock().unwrap().clone()
}
#[tauri::command]
fn save_config(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    config: DesktopConfig,
) -> tauri::Result<()> {
    let dir = state_dir(&app)?;
    let previous_runtime = state.runtime.lock().unwrap().take();
    drop(previous_runtime);
    let rt = match build_runtime(&config, dir) {
        Ok(runtime) => runtime,
        Err(runtime_error) => {
            let message = runtime_error.to_string();
            tracing::error!(error = %message, "failed to apply desktop configuration");
            *state.runtime_error.lock().unwrap() = Some(message);
            return Err(runtime_error);
        }
    };
    persist_config(&app, &config)?;
    *state.config.lock().unwrap() = config;
    *state.runtime.lock().unwrap() = Some(rt);
    *state.runtime_error.lock().unwrap() = None;
    Ok(())
}
#[tauri::command]
async fn list_threads(state: State<'_, AppState>) -> tauri::Result<Vec<Thread>> {
    let Some(rt) = state
        .runtime
        .lock()
        .unwrap()
        .as_ref()
        .map(|x| (x.store.clone(), x.state_dir.clone()))
    else {
        return Ok(vec![]);
    };
    let dir = rt.1.join("threads");
    let mut out = Vec::new();
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    while let Some(e) = entries.next_entry().await.map_err(error)? {
        if e.path().extension().and_then(|x| x.to_str()) == Some("json") {
            if let Ok(raw) = tokio::fs::read_to_string(e.path()).await {
                if let Ok(t) = serde_json::from_str(&raw) {
                    out.push(t)
                }
            }
        }
    }
    out.sort_by(|a: &Thread, b: &Thread| b.updated_at.cmp(&a.updated_at));
    let _ = rt.0;
    Ok(out)
}
#[tauri::command]
async fn create_thread(state: State<'_, AppState>, workspace: String) -> tauri::Result<String> {
    let Some(rt) = state
        .runtime
        .lock()
        .unwrap()
        .as_ref()
        .map(|x| (x.store.clone(), x.state_dir.clone()))
    else {
        return Err(error("请先完成设置"));
    };
    let id = new_id("thread");
    let mut t = Thread::new(&id);
    t.metadata = json!({"workspace":workspace});
    rt.0.compare_and_commit(t, &lite_agent_runtime::LeaseFence::from_bytes([]))
        .await
        .map_err(error)?;
    let _ = rt.1;
    Ok(id)
}
#[tauri::command]
async fn run_turn(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    thread_id: String,
    user_text: String,
) -> tauri::Result<()> {
    let (agent, trace_collector) = state
        .runtime
        .lock()
        .unwrap()
        .as_ref()
        .map(|x| (x.agent.clone(), x.trace_collector.clone()))
        .ok_or_else(|| error("请先完成设置"))?;
    tracing::info!(thread_id, "desktop turn command started");
    let result = agent
        .run_turn_stream(&thread_id, user_text, move |event| {
            let _ = app.emit("turn-event", desktop_turn_event(event));
        })
        .await;
    trace_collector.flush().await;
    match result {
        Ok(outcome) => {
            tracing::info!(thread_id, ?outcome, "desktop turn command finished");
            Ok(())
        }
        Err(turn_error) => {
            tracing::error!(thread_id, error = %turn_error, "desktop turn command failed");
            Err(error(turn_error))
        }
    }
}

#[tauri::command]
fn abort_turn(state: State<'_, AppState>, thread_id: String) -> tauri::Result<()> {
    let agent = state
        .runtime
        .lock()
        .unwrap()
        .as_ref()
        .map(|runtime| runtime.agent.clone())
        .ok_or_else(|| error("请先完成设置"))?;
    agent.abort(&thread_id).map_err(error)
}

fn desktop_turn_event(event: TurnStreamEvent) -> Value {
    match event {
        TurnStreamEvent::Model(model) => match model {
            TurnModelEvent::RequestStarted { iteration } => {
                json!({"type":"model_iteration_started","iteration":iteration})
            }
            TurnModelEvent::AssistantMessage { text } => {
                json!({"type":"assistant_message","text":text})
            }
            TurnModelEvent::AssistantDelta { text } => {
                json!({"type":"assistant_delta","text":text})
            }
        },
        TurnStreamEvent::State(state) => desktop_state_event(state),
        TurnStreamEvent::Runtime(RuntimeEvent {
            source,
            message,
            metadata,
        }) => json!({"type":"runtime","source":source,"message":message,"metadata":metadata}),
    }
}

fn desktop_state_event(event: TurnStateEvent) -> Value {
    match event {
        TurnStateEvent::TurnStarted { thread_id, turn_id } => {
            json!({"type":"turn_started","thread_id":thread_id,"turn_id":turn_id})
        }
        TurnStateEvent::FunctionCallsRequested { calls } => {
            json!({"type":"tool_calls_requested","calls":calls})
        }
        TurnStateEvent::FunctionStarted { call_id, name } => {
            json!({"type":"tool_started","call_id":call_id,"name":name})
        }
        TurnStateEvent::FunctionCompleted { call_id, name } => {
            json!({"type":"tool_completed","call_id":call_id,"name":name})
        }
        TurnStateEvent::FunctionFailed {
            call_id,
            name,
            error,
        } => json!({"type":"tool_failed","call_id":call_id,"name":name,"error":error}),
        TurnStateEvent::Suspended { suspension } => {
            json!({"type":"turn_suspended","suspension":suspension})
        }
        TurnStateEvent::TurnFinished { outcome } => match outcome {
            TurnOutcome::AssistantMessage { text } => {
                json!({"type":"turn_finished","outcome":"assistant_message","text":text})
            }
            TurnOutcome::Suspended { suspension } => {
                json!({"type":"turn_finished","outcome":"suspended","suspension":suspension})
            }
            TurnOutcome::Failed { error } => {
                json!({"type":"turn_finished","outcome":"failed","error":error})
            }
            TurnOutcome::Aborted { reason } => {
                json!({"type":"turn_finished","outcome":"aborted","reason":reason})
            }
        },
        TurnStateEvent::TurnFailed { error } => json!({"type":"turn_failed","error":error}),
        TurnStateEvent::TurnAborted { reason } => json!({"type":"turn_aborted","reason":reason}),
        TurnStateEvent::TurnTokenUsage { usage } => {
            json!({"type":"token_usage","usage":usage})
        }
    }
}
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let diagnostics_dir = state_dir(app.handle())?;
            let logging_guard = init_file_logging(&diagnostics_dir).map_err(error)?;
            let c = load_config(app.handle()).unwrap_or_default();
            let (runtime, runtime_error) = if c.base_url.is_empty() || c.model.is_empty() {
                (None, None)
            } else {
                match build_runtime(&c, diagnostics_dir.clone()) {
                    Ok(runtime) => (Some(runtime), None),
                    Err(runtime_error) => {
                        let message = runtime_error.to_string();
                        tracing::error!(error = %runtime_error, "desktop runtime initialization failed");
                        (None, Some(message))
                    }
                }
            };
            app.manage(AppState {
                config: Mutex::new(c),
                runtime: Mutex::new(runtime),
                runtime_error: Mutex::new(runtime_error),
                _logging_guard: logging_guard,
                diagnostics_dir,
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
            get_diagnostics_dir,
            get_runtime_error,
            save_config,
            list_threads,
            create_thread,
            run_turn,
            abort_turn
        ])
        .run(tauri::generate_context!())
        .expect("error while running lite-agent");
}

#[cfg(test)]
mod tests {
    use super::{DesktopConfig, DEFAULT_MAX_CONTEXT_TOKENS, DEFAULT_MAX_MODEL_ITERATIONS};

    #[test]
    fn legacy_desktop_config_receives_long_task_defaults() {
        let config: DesktopConfig = serde_json::from_str(
            r#"{
                "base_url": "https://example.com/v1",
                "model": "example-model",
                "api_key": "secret",
                "workspace": "C:\\\\workspace"
            }"#,
        )
        .expect("legacy config");

        assert_eq!(config.max_context_tokens, DEFAULT_MAX_CONTEXT_TOKENS);
        assert_eq!(config.max_model_iterations, DEFAULT_MAX_MODEL_ITERATIONS);
        assert!(config.qianfan_api_key.is_empty());
    }
}
