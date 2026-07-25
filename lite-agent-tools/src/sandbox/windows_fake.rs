//! Windows MVP backend. It preserves the requested policy in the request and
//! validates the workspace, but does not provide OS-level isolation yet.
//! A future implementation should replace this with an AppContainer/job
//! object backend before the app is marketed as a secure sandbox.

use super::*;
use std::process::Stdio;
use std::time::Instant;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

#[derive(Debug, Default, Clone, Copy)]
pub struct WindowsFakeBackend;

impl WindowsFakeBackend {
    pub fn new() -> Self {
        Self
    }
}

impl SandboxBackend for WindowsFakeBackend {
    fn name(&self) -> &str {
        "windows-fake"
    }

    fn resolve_policy(&self, policy: &SandboxPolicy) -> SandboxResult<SandboxPolicyResolution> {
        policy.validate()?;
        Ok(SandboxPolicyResolution {
            requested: policy.clone(),
            effective: EffectiveSandboxPolicy {
                filesystem: policy.filesystem.requested.clone(),
                network: policy.network.requested,
                process: policy.process.requested,
                identity: policy.identity.requested,
            },
            warnings: vec![SandboxWarning {
                dimension: SandboxPolicyDimension::Filesystem,
                message: "Windows MVP 使用 fake sandbox；当前不提供 OS 级文件系统隔离。"
                    .to_string(),
            }],
        })
    }

    fn execute<'a>(
        &'a self,
        request: SandboxRequest,
    ) -> Pin<Box<dyn Future<Output = SandboxResult<SandboxOutput>> + Send + 'a>> {
        Box::pin(async move {
            request.validate()?;
            let started = Instant::now();
            let mut child = Command::new(&request.program)
                .args(&request.args)
                .current_dir(&request.cwd)
                .envs(&request.environment)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?;
            let mut stdout = child.stdout.take().expect("stdout piped");
            let mut stderr = child.stderr.take().expect("stderr piped");
            let (out, err, status) = tokio::select! {
                _ = request.cancellation.cancelled() => {
                    let _ = child.kill().await;
                    (Vec::new(), Vec::new(), SandboxStatus::Cancelled)
                }
                result = async {
                    let mut out = Vec::new(); let mut err = Vec::new();
                    stdout.read_to_end(&mut out).await?;
                    stderr.read_to_end(&mut err).await?;
                    Ok::<_, std::io::Error>((out, err, child.wait().await?))
                } => {
                    let (out, err, status) = result?;
                    let code = status.code();
                    (out, err, SandboxStatus::Exited { code })
                }
            };
            Ok(SandboxOutput {
                stdout: out,
                stderr: err,
                status,
                duration: started.elapsed(),
                stdout_truncated: false,
                stderr_truncated: false,
                warnings: vec![SandboxWarning {
                    dimension: SandboxPolicyDimension::Filesystem,
                    message: "fake sandbox: OS 隔离未启用".to_string(),
                }],
            })
        })
    }
}
