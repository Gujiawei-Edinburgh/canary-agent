use canary_agent_runtime::{AgentError, AudioEncoding, AudioFormat, Result};
use std::time::Duration;
use url::Url;

#[derive(Clone)]
pub struct RealtimeConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub voice: String,
    pub transcription_model: String,
    pub connect_timeout: Duration,
}

impl std::fmt::Debug for RealtimeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealtimeConfig")
            .field("model", &self.model)
            .field("voice", &self.voice)
            .field("api_key", &"[redacted]")
            .finish_non_exhaustive()
    }
}

impl RealtimeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.api_key.is_empty() || self.model.is_empty() {
            return Err(AgentError::Model(
                "realtime API key and model are required".into(),
            ));
        }
        self.websocket_url().map(|_| ())
    }
    pub fn new(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            model: model.into(),
            voice: "marin".into(),
            transcription_model: "gpt-4o-mini-transcribe".into(),
            connect_timeout: Duration::from_secs(20),
        }
    }

    pub fn input_format(&self) -> AudioFormat {
        AudioFormat {
            encoding: AudioEncoding::PcmS16Le,
            sample_rate: 24000,
            channels: 1,
        }
    }

    pub fn output_format(&self) -> AudioFormat {
        AudioFormat {
            encoding: AudioEncoding::PcmS16Le,
            sample_rate: 24000,
            channels: 1,
        }
    }

    pub(super) fn websocket_url(&self) -> Result<Url> {
        let mut url = Url::parse(&self.endpoint)
            .map_err(|_| AgentError::Model("invalid realtime endpoint URL".into()))?;
        let scheme = match url.scheme() {
            "https" | "wss" => "wss",
            "http" | "ws" => "ws",
            _ => {
                return Err(AgentError::Model(
                    "realtime endpoint requires http(s) or ws(s)".into(),
                ))
            }
        };
        url.set_scheme(scheme)
            .map_err(|_| AgentError::Model("invalid realtime URL scheme".into()))?;
        if !url.username().is_empty() || url.password().is_some() {
            return Err(AgentError::Model(
                "realtime endpoint must not contain credentials".into(),
            ));
        }
        if matches!(
            url.path().trim_end_matches('/'),
            "" | "/v1" | "/compatible-mode/v1"
        ) {
            url.set_path("/v1/realtime");
        }
        let existing = url
            .query_pairs()
            .find(|(key, _)| key == "model")
            .map(|(_, value)| value.into_owned());
        if let Some(existing) = existing {
            if existing != self.model {
                return Err(AgentError::Model(
                    "endpoint model differs from configured realtime model".into(),
                ));
            }
        } else {
            url.query_pairs_mut().append_pair("model", &self.model);
        }
        Ok(url)
    }
}
