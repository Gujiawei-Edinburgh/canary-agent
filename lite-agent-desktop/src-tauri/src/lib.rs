use lite_agent_kernel::events::{new_id, Thread};
use lite_agent_openai::{ChatCompletionsClient, ModelConfig};
use lite_agent_runtime::{
    builtin_registry, Agent, AgentConfig, FunctionContext, LocalSessionCoordinator, Result,
    ThreadStore,
};
use lite_agent_store_json::JsonFileThreadStore;
use lite_agent_tools::sandbox::{SandboxBackend, SandboxPolicy};
use lite_agent_tools::{
    register_time_tools, register_web_search, AuthorizationDecision, ExecAuthorizer,
    ExecCommandConfig, ExecCommandTool, WebSearchConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tauri::{Emitter, Manager, State};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DesktopConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    pub workspace: String,
}

struct AppState {
    config: Mutex<DesktopConfig>,
    runtime: Mutex<Option<Runtime>>,
}
struct Runtime {
    agent: Agent,
    store: Arc<JsonFileThreadStore>,
    state_dir: PathBuf,
}

fn error(message: impl ToString) -> tauri::Error {
    tauri::Error::Setup(
        (Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            message.to_string(),
        )) as Box<dyn std::error::Error>)
            .into(),
    )
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
    let store = Arc::new(JsonFileThreadStore::open(&dir).map_err(error)?);
    let model = Arc::new(ChatCompletionsClient::new(ModelConfig {
        base_url: c.base_url.clone(),
        api_key: c.api_key.clone(),
        model: c.model.clone(),
        reasoning_effort: ModelConfig::default_reasoning_effort(),
    }));
    let mut reg = builtin_registry();
    register_time_tools(&mut reg);
    register_web_search(&mut reg, WebSearchConfig::default());
    reg.register(ExecCommandTool::new(
        ExecCommandConfig::new(
            PathBuf::from(&c.workspace),
            backend(),
            SandboxPolicy::workspace_read_write_with_host_network(&c.workspace),
            Arc::new(DenyAuthorizer),
        )
        .with_workspace_resolver(WorkspaceResolver {
            fallback: PathBuf::from(&c.workspace),
        }),
    ));
    let agent=Agent::new(AgentConfig{system_prompt:"你是 lite-agent，使用中文回答。需要执行命令时使用 exec_command，需要查询公开网页时使用 web_search，需要当前时间时使用 get_current_time。".into(),..AgentConfig::default()},store.clone(),model,reg,Arc::new(LocalSessionCoordinator::default()));
    Ok(Runtime {
        agent,
        store,
        state_dir: dir,
    })
}

#[tauri::command]
fn get_config(state: State<'_, AppState>) -> DesktopConfig {
    state.config.lock().unwrap().clone()
}
#[tauri::command]
fn save_config(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    config: DesktopConfig,
) -> tauri::Result<()> {
    persist_config(&app, &config)?;
    let dir = state_dir(&app)?;
    let rt = build_runtime(&config, dir)?;
    *state.config.lock().unwrap() = config;
    *state.runtime.lock().unwrap() = Some(rt);
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
    let mut entries = tokio::fs::read_dir(dir).await.map_err(error)?;
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
    let agent = state
        .runtime
        .lock()
        .unwrap()
        .as_ref()
        .map(|x| x.agent.clone())
        .ok_or_else(|| error("请先完成设置"))?;
    agent
        .run_turn_stream(&thread_id, user_text, move |event| {
            let message = format!("{event:?}");
            let _ = app.emit("turn-event", json!({"kind":"runtime","message":message}));
        })
        .await
        .map(|_| ())
        .map_err(error)
}
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let c = load_config(app.handle())?;
            let dir = state_dir(app.handle())?;
            let runtime = if c.base_url.is_empty() || c.model.is_empty() {
                None
            } else {
                Some(build_runtime(&c, dir)?)
            };
            app.manage(AppState {
                config: Mutex::new(c),
                runtime: Mutex::new(runtime),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
            save_config,
            list_threads,
            create_thread,
            run_turn
        ])
        .run(tauri::generate_context!())
        .expect("error while running lite-agent");
}
