//! Semantic contracts for native command sandbox backends.
//!
//! This crate does not implement an operating-system sandbox. It defines the
//! isolation guarantees requested by `exec_command` and the execution boundary
//! used by native, container, or user-provided backends.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::Notify;

mod seatbelt;

pub use seatbelt::MacOsSeatbeltBackend;

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "linux")]
pub use linux::LinuxNativeBackend;

pub type SandboxResult<T> = Result<T, SandboxError>;

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("invalid sandbox request: {0}")]
    InvalidRequest(String),

    #[error("sandbox policy is unsupported: {0}")]
    UnsupportedPolicy(String),

    #[error("sandbox launch failed: {0}")]
    Launch(String),

    #[error("sandbox I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A backend that executes a native command under a semantic isolation policy.
pub trait SandboxBackend: Send + Sync {
    fn name(&self) -> &str;

    /// Validate whether a request can be launched under its requested policy.
    ///
    /// This is the only phase that may produce an approval suspension. Once
    /// execution begins, a policy violation is a failed, single-attempt
    /// execution and must not be retried automatically.
    fn preflight(&self, request: &SandboxRequest) -> SandboxResult<SandboxPreflight> {
        request.validate()?;
        self.resolve_policy(&request.policy)?;
        Ok(SandboxPreflight::Allowed)
    }

    /// Resolve the requested guarantees into the policy this backend can
    /// actually enforce.
    ///
    /// The returned warnings must describe every requested guarantee that was
    /// weakened or changed by fallback.
    fn resolve_policy(&self, policy: &SandboxPolicy) -> SandboxResult<SandboxPolicyResolution>;

    fn execute<'a>(
        &'a self,
        request: SandboxRequest,
    ) -> Pin<Box<dyn Future<Output = SandboxResult<SandboxOutput>> + Send + 'a>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxPreflight {
    Allowed,
    PolicyViolation { reason: String },
}

/// A cancellation signal shared by the agent runtime and a sandbox backend.
///
/// Backends should use this signal to terminate the complete command process
/// tree, not only the direct child process.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        let notified = self.notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if !self.is_cancelled() {
            notified.await;
        }
    }
}

#[derive(Clone, Debug)]
pub struct SandboxRequest {
    /// The requested executable path on the host.
    pub program: PathBuf,
    pub args: Vec<String>,
    /// The requested working directory on the host.
    pub cwd: PathBuf,
    pub environment: BTreeMap<String, String>,
    pub cancellation: CancellationToken,
    pub policy: SandboxPolicy,
}

impl SandboxRequest {
    pub fn validate(&self) -> SandboxResult<()> {
        if self.program.as_os_str().is_empty() {
            return Err(SandboxError::InvalidRequest(
                "program must not be empty".to_string(),
            ));
        }
        if !self.cwd.is_absolute() {
            return Err(SandboxError::InvalidRequest(
                "working directory must be an absolute host path".to_string(),
            ));
        }
        self.policy.validate()
    }
}

/// Semantic isolation requirements for one command execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxPolicy {
    pub filesystem: PolicySetting<FilesystemPolicy>,
    pub network: PolicySetting<NetworkAccess>,
    pub process: PolicySetting<ProcessPolicy>,
    pub identity: PolicySetting<IdentityIsolation>,
    pub kernel_ops: PolicySetting<KernelOpsPolicy>,
}

impl SandboxPolicy {
    pub fn workspace_read_only(host_path: impl Into<PathBuf>) -> Self {
        Self {
            filesystem: PolicySetting::strict(FilesystemPolicy::workspace(
                host_path,
                FilesystemAccess::ReadOnly,
            )),
            ..Self::default()
        }
    }

    pub fn workspace_read_write(host_path: impl Into<PathBuf>) -> Self {
        Self {
            filesystem: PolicySetting::strict(FilesystemPolicy::workspace(
                host_path,
                FilesystemAccess::ReadWrite,
            )),
            ..Self::default()
        }
    }

    pub fn workspace_read_write_with_host_network(host_path: impl Into<PathBuf>) -> Self {
        Self {
            filesystem: PolicySetting::strict(FilesystemPolicy::workspace(
                host_path,
                FilesystemAccess::ReadWrite,
            )),
            network: PolicySetting::strict(NetworkAccess::Host),
            ..Self::default()
        }
    }

    pub fn workspace_scoped(
        base: impl Into<PathBuf>,
        visible: impl Into<PathBuf>,
        access: FilesystemAccess,
    ) -> SandboxResult<Self> {
        Ok(Self {
            filesystem: PolicySetting::strict(FilesystemPolicy::workspace_scoped(
                base, visible, access,
            )?),
            ..Self::default()
        })
    }

    pub fn validate(&self) -> SandboxResult<()> {
        self.filesystem.requested.validate()
    }
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            filesystem: PolicySetting::strict(FilesystemPolicy::default()),
            network: PolicySetting::strict(NetworkAccess::Isolated),
            process: PolicySetting::fallback(ProcessPolicy::default()),
            identity: PolicySetting::fallback(IdentityIsolation::Unprivileged),
            kernel_ops: PolicySetting::fallback(KernelOpsPolicy::baseline()),
        }
    }
}

/// Controls what happens when a backend cannot provide one requested
/// isolation guarantee.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UnsupportedPolicyBehavior {
    #[default]
    Error,
    WarnAndFallback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicySetting<T> {
    pub requested: T,
    pub unsupported: UnsupportedPolicyBehavior,
}

impl<T> PolicySetting<T> {
    pub fn strict(requested: T) -> Self {
        Self {
            requested,
            unsupported: UnsupportedPolicyBehavior::Error,
        }
    }

    pub fn fallback(requested: T) -> Self {
        Self {
            requested,
            unsupported: UnsupportedPolicyBehavior::WarnAndFallback,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxPolicyResolution {
    pub requested: SandboxPolicy,
    pub effective: EffectiveSandboxPolicy,
    pub warnings: Vec<SandboxWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveSandboxPolicy {
    pub filesystem: FilesystemPolicy,
    pub network: NetworkAccess,
    pub process: ProcessPolicy,
    pub identity: IdentityIsolation,
    pub kernel_ops: KernelOpsPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxPolicyDimension {
    Filesystem,
    Network,
    Process,
    Identity,
    KernelOps,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxWarning {
    pub dimension: SandboxPolicyDimension,
    pub message: String,
}

/// High-level operations that can affect the host kernel or other processes.
///
/// This is deliberately independent of syscall names and numbers. Backends
/// translate these operations into their native policy language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KernelOp {
    MountFilesystem,
    ChangeNamespace,
    TraceProcess,
    AccessProcessMemory,
    UseKernelInstrumentation,
    LoadKernelCode,
    AccessRawDevice,
    ManageSystemPower,
}

impl KernelOp {
    #[cfg(target_os = "linux")]
    fn all() -> &'static [Self] {
        &[
            Self::MountFilesystem,
            Self::ChangeNamespace,
            Self::TraceProcess,
            Self::AccessProcessMemory,
            Self::UseKernelInstrumentation,
            Self::LoadKernelCode,
            Self::AccessRawDevice,
            Self::ManageSystemPower,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelOpViolationAction {
    ReturnPermissionDenied,
    KillProcess,
}

/// Kernel-operation restrictions requested from a sandbox backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelOpsPolicy {
    /// Do not install a kernel-operation filter.
    Unrestricted,
    /// Deny the listed operation groups and allow all other operations.
    DenyList {
        denied: BTreeSet<KernelOp>,
        violation: KernelOpViolationAction,
    },
    /// Allow only the listed operation groups. This is intentionally opt-in:
    /// arbitrary shells and language runtimes need a broad syscall surface.
    AllowList {
        allowed: BTreeSet<KernelOp>,
        violation: KernelOpViolationAction,
    },
}

impl KernelOpsPolicy {
    /// A practical baseline for untrusted command execution. It blocks
    /// operations that can create further isolation escapes, inspect or alter
    /// other processes, load kernel functionality, access raw device handles,
    /// or change system power state.
    pub fn baseline() -> Self {
        Self::DenyList {
            denied: [
                KernelOp::MountFilesystem,
                KernelOp::ChangeNamespace,
                KernelOp::TraceProcess,
                KernelOp::AccessProcessMemory,
                KernelOp::UseKernelInstrumentation,
                KernelOp::LoadKernelCode,
                KernelOp::AccessRawDevice,
                KernelOp::ManageSystemPower,
            ]
            .into_iter()
            .collect(),
            violation: KernelOpViolationAction::ReturnPermissionDenied,
        }
    }
}

/// Filesystem visibility and write access requested by the command.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum FilesystemPolicy {
    /// No host filesystem access other than what the backend needs to launch
    /// the command.
    #[default]
    Isolated,
    /// Expose only one subtree of a managed workspace. Paths outside the
    /// base remain read-only, while sibling paths below the base are hidden.
    WorkspaceScoped {
        base: PathBuf,
        visible: PathBuf,
        access: FilesystemAccess,
    },
    /// The command can access the host filesystem without filesystem
    /// isolation. This must be selected explicitly.
    Host,
}

impl FilesystemPolicy {
    pub fn workspace(host_path: impl Into<PathBuf>, access: FilesystemAccess) -> Self {
        let host_path = host_path.into();
        Self::WorkspaceScoped {
            base: host_path.clone(),
            visible: host_path,
            access,
        }
    }

    pub fn workspace_scoped(
        base: impl Into<PathBuf>,
        visible: impl Into<PathBuf>,
        access: FilesystemAccess,
    ) -> SandboxResult<Self> {
        let base = normalize_absolute_path(&base.into())?;
        let visible = normalize_absolute_path(&visible.into())?;
        if !visible.starts_with(&base) {
            return Err(SandboxError::InvalidRequest(format!(
                "visible workspace path {} must be inside base path {}",
                visible.display(),
                base.display()
            )));
        }
        Ok(Self::WorkspaceScoped {
            base,
            visible,
            access,
        })
    }

    /// Resolve the effective access for an absolute path using longest-prefix
    /// matching on complete path components.
    pub fn access_for(&self, path: impl AsRef<std::path::Path>) -> SandboxResult<FilesystemAccess> {
        let path = normalize_absolute_path(path.as_ref())?;
        match self {
            Self::Isolated => Ok(FilesystemAccess::Denied),
            Self::Host => Ok(FilesystemAccess::ReadWrite),
            Self::WorkspaceScoped {
                base,
                visible,
                access,
            } => {
                self.validate()?;
                if path.starts_with(visible) {
                    Ok(*access)
                } else if path.starts_with(base) {
                    Ok(FilesystemAccess::Denied)
                } else {
                    Ok(FilesystemAccess::ReadOnly)
                }
            }
        }
    }

    fn validate(&self) -> SandboxResult<()> {
        match self {
            Self::WorkspaceScoped {
                base,
                visible,
                access: _,
            } => {
                let base = normalize_absolute_path(base)?;
                let visible = normalize_absolute_path(visible)?;
                if !visible.starts_with(&base) {
                    return Err(SandboxError::InvalidRequest(format!(
                        "visible workspace path {} must be inside base path {}",
                        visible.display(),
                        base.display()
                    )));
                }
            }
            Self::Isolated | Self::Host => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilesystemAccess {
    Denied,
    ReadOnly,
    ReadWrite,
}

fn normalize_absolute_path(path: &std::path::Path) -> SandboxResult<PathBuf> {
    if !path.is_absolute() {
        return Err(SandboxError::InvalidRequest(
            "filesystem paths must be absolute".to_string(),
        ));
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::RootDir => normalized.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    return Err(SandboxError::InvalidRequest(format!(
                        "filesystem path escapes its root: {}",
                        path.display()
                    )));
                }
            }
            std::path::Component::Normal(value) => normalized.push(value),
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
        }
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::{FilesystemAccess, FilesystemPolicy, KernelOp, KernelOpsPolicy};
    use std::collections::BTreeSet;

    #[test]
    fn scoped_workspace_resolves_visibility_by_path() {
        let policy = FilesystemPolicy::workspace_scoped(
            "/workspace",
            "/workspace/user-a/thread-a",
            FilesystemAccess::ReadWrite,
        )
        .expect("valid scoped workspace");

        assert_eq!(
            policy
                .access_for("/workspace/user-a/thread-a/file")
                .unwrap(),
            FilesystemAccess::ReadWrite
        );
        assert_eq!(
            policy
                .access_for("/workspace/user-a/thread-b/file")
                .unwrap(),
            FilesystemAccess::Denied
        );
        assert_eq!(
            policy.access_for("/etc/hosts").unwrap(),
            FilesystemAccess::ReadOnly
        );
    }

    #[test]
    fn baseline_blocks_high_risk_kernel_operations() {
        let KernelOpsPolicy::DenyList { denied, .. } = KernelOpsPolicy::baseline() else {
            panic!("baseline must be a deny list");
        };
        assert_eq!(denied.len(), 8);
        assert!(denied.contains(&KernelOp::ChangeNamespace));
        assert!(denied.contains(&KernelOp::TraceProcess));
        assert!(denied.contains(&KernelOp::LoadKernelCode));
    }

    #[test]
    fn explicit_kernel_operation_lists_are_preserved() {
        let allowed = BTreeSet::from([KernelOp::TraceProcess]);
        let policy = KernelOpsPolicy::AllowList {
            allowed: allowed.clone(),
            violation: super::KernelOpViolationAction::KillProcess,
        };
        assert_eq!(
            policy,
            KernelOpsPolicy::AllowList {
                allowed,
                violation: super::KernelOpViolationAction::KillProcess,
            }
        );
    }
}

/// Network visibility requested by the command.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NetworkAccess {
    /// The command cannot use networking.
    Denied,
    /// The command runs in a separate network environment. It cannot access
    /// the host network, but the backend may provide isolated loopback or
    /// other explicitly configured connectivity.
    #[default]
    Isolated,
    /// The command shares the host network, including access to services such
    /// as a developer-configured loopback proxy.
    Host,
}

/// Process visibility and descendant cleanup are independent guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessPolicy {
    pub visibility: ProcessVisibility,
    pub terminate_descendants: bool,
}

impl Default for ProcessPolicy {
    fn default() -> Self {
        Self {
            visibility: ProcessVisibility::Isolated,
            terminate_descendants: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ProcessVisibility {
    #[default]
    Isolated,
    Host,
}

/// Whether the command inherits the caller's host identity or runs with a
/// backend-selected unprivileged identity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum IdentityIsolation {
    #[default]
    Unprivileged,
    Host,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxOutput {
    pub status: SandboxStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub warnings: Vec<SandboxWarning>,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub duration: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxStatus {
    Exited { code: i32 },
    Signaled { signal: i32 },
    TimedOut,
    Cancelled,
    PolicyViolation { reason: String },
}

pub(crate) fn classify_policy_violation(status: SandboxStatus, stderr: &[u8]) -> SandboxStatus {
    if !matches!(status, SandboxStatus::Exited { .. }) {
        return status;
    }
    let message = String::from_utf8_lossy(stderr);
    let lower = message.to_ascii_lowercase();
    if lower.contains("operation not permitted") || lower.contains("permission denied") {
        return SandboxStatus::PolicyViolation {
            reason: message.trim().to_string(),
        };
    }
    status
}
