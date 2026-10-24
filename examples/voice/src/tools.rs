use canary_agent_runtime::FunctionRegistry;
use canary_agent_runtime::{FunctionContext, Result};
use canary_agent_tools::sandbox::{SandboxBackend, SandboxPolicy};
use canary_agent_tools::{
    AuthorizationDecision, ExecAuthorizer, ExecCommandConfig, ExecCommandTool,
};
use canary_agent_tools::{ExaWebSearchConfig, register_time_tools, register_web_search_tools};
use std::{future::Future, path::PathBuf, pin::Pin, sync::Arc};

pub fn registry(workspace: PathBuf) -> FunctionRegistry {
    let mut registry = FunctionRegistry::new();
    register_time_tools(&mut registry);
    if let Ok(key) = std::env::var("EXA_API_KEY") {
        register_web_search_tools(&mut registry, ExaWebSearchConfig::new(key));
    }
    registry.register(ExecCommandTool::new(ExecCommandConfig::new(
        workspace.clone(),
        backend(),
        SandboxPolicy::workspace_read_write_with_host_network(workspace),
        Arc::new(DenyEscalation),
    )));
    registry
}

struct DenyEscalation;
impl ExecAuthorizer for DenyEscalation {
    fn authorize<'a>(
        &'a self,
        _request: &'a canary_agent_tools::ExecRequest,
        _policy: &'a SandboxPolicy,
        _violation: &'a str,
        _context: &'a FunctionContext,
    ) -> Pin<Box<dyn Future<Output = Result<AuthorizationDecision>> + Send + 'a>> {
        Box::pin(async {
            Ok(AuthorizationDecision::Deny {
                reason: "the voice example does not grant sandbox escalation".into(),
            })
        })
    }
}

#[cfg(target_os = "macos")]
fn backend() -> Arc<dyn SandboxBackend> {
    Arc::new(canary_agent_tools::sandbox::MacOsSeatbeltBackend::new())
}
#[cfg(target_os = "linux")]
fn backend() -> Arc<dyn SandboxBackend> {
    Arc::new(canary_agent_tools::sandbox::LinuxNativeBackend::new())
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn backend() -> Arc<dyn SandboxBackend> {
    compile_error!("the voice exec tool requires macOS or Linux");
}
