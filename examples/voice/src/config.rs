use crate::model_bridge::BridgeConfig;
use canary_agent_runtime::{AgentError, Result};
use std::{env, path::PathBuf};

pub struct VoiceConfig {
    pub model: BridgeConfig,
    pub state_dir: PathBuf,
    pub thread_id: String,
    pub workspace: PathBuf,
}

impl VoiceConfig {
    pub fn load(provider: &str) -> Result<Self> {
        let _ = dotenvy::dotenv();
        let _ = dotenvy::from_path(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".env"));
        let endpoint = if provider == "moss" {
            env::var("MOSS_ENDPOINT").unwrap_or_else(|_| "wss://api.mosi.cn/v1/realtime".into())
        } else {
            env::var("REAL_TIME_ENDPOINT").map_err(|_| {
                AgentError::Model(
                    "set REAL_TIME_ENDPOINT in examples/voice/.env or the environment".into(),
                )
            })?
        };
        let key_name = if provider == "moss" {
            "MOSS_API_KEY"
        } else {
            "REAL_TIME_API_KEY"
        };
        let api_key =
            env::var(key_name).map_err(|_| AgentError::Model(format!("set {key_name}")))?;
        let mut model = BridgeConfig {
            endpoint,
            api_key,
            model: if provider == "moss" {
                env::var("MOSS_MODEL").unwrap_or_else(|_| "moss-live-1.0".into())
            } else {
                env::var("REAL_TIME_MODEL").unwrap_or_else(|_| "qwen3.8-omni-flash-realtime".into())
            },
            voice: env::var("REAL_TIME_VOICE").unwrap_or_else(|_| "Tina".into()),
            python: env::var("VOICE_PYTHON").unwrap_or_else(|_| {
                let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".venv/bin/python");
                if local.is_file() {
                    local.to_string_lossy().into_owned()
                } else {
                    "python3".into()
                }
            }),
            script: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(if provider == "moss" {
                "python/moss_bridge.py"
            } else {
                "python/worker.py"
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
