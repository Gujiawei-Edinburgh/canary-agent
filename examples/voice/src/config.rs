use crate::qwen::QwenConfig;
use canary_agent_runtime::{AgentError, Result};
use std::{env, path::PathBuf};

pub struct VoiceConfig {
    pub model: QwenConfig,
    pub state_dir: PathBuf,
    pub thread_id: String,
    pub workspace: PathBuf,
}

impl VoiceConfig {
    pub fn load() -> Result<Self> {
        let _ = dotenvy::dotenv();
        let _ = dotenvy::from_path(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".env"));
        let endpoint = env::var("REAL_TIME_ENDPOINT").map_err(|_| {
            AgentError::Model(
                "set REAL_TIME_ENDPOINT in examples/voice/.env or the environment".into(),
            )
        })?;
        let api_key = env::var("REAL_TIME_API_KEY")
            .map_err(|_| AgentError::Model("set REAL_TIME_API_KEY".into()))?;
        if env::var("REAL_TIME_PROTOCOL").is_ok_and(|value| value != "qwen") {
            return Err(AgentError::Model("the voice example uses Qwen's Python SDK; REAL_TIME_PROTOCOL must be qwen or unset".into()));
        }
        let mut model = QwenConfig {
            endpoint,
            api_key,
            model: env::var("REAL_TIME_MODEL")
                .unwrap_or_else(|_| "qwen3.8-omni-flash-realtime".into()),
            voice: env::var("REAL_TIME_VOICE").unwrap_or_else(|_| "Tina".into()),
            python: env::var("VOICE_PYTHON").unwrap_or_else(|_| {
                let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".venv/bin/python");
                if local.is_file() {
                    local.to_string_lossy().into_owned()
                } else {
                    "python3".into()
                }
            }),
        };
        model.validate()?;
        Ok(Self {
            model,
            state_dir: env::var_os("VOICE_STATE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(".canary-agent-voice")),
            thread_id: env::var("VOICE_THREAD_ID")
                .unwrap_or_else(|_| canary_agent_kernel::new_id("voice")),
            workspace: env::var_os("VOICE_WORKSPACE")
                .map(PathBuf::from)
                .unwrap_or(env::current_dir()?),
        })
    }
}
