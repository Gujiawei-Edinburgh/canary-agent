//! Provider-neutral JSONL messages shared by the voice application's bridges.
use canary_agent_runtime::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::{path::Path, process::Stdio, sync::Arc};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
};

struct Process {
    _child: Child,
}

pub struct BridgeSender {
    input: BufWriter<ChildStdin>,
    _process: Arc<Process>,
}

pub struct BridgeReceiver {
    output: Lines<BufReader<ChildStdout>>,
    _process: Arc<Process>,
}

pub fn spawn(python: &str, script: &Path) -> Result<(BridgeSender, BridgeReceiver)> {
    let mut child = Command::new(python)
        .arg("-u")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| AgentError::Model(format!("cannot start realtime bridge: {e}")))?;
    let input = child.stdin.take().expect("piped stdin");
    let output = child.stdout.take().expect("piped stdout");
    let process = Arc::new(Process { _child: child });
    Ok((
        BridgeSender {
            input: BufWriter::new(input),
            _process: process.clone(),
        },
        BridgeReceiver {
            output: BufReader::new(output).lines(),
            _process: process,
        },
    ))
}

impl BridgeSender {
    pub async fn write(&mut self, value: &impl Serialize) -> Result<()> {
        let mut data = serde_json::to_vec(value)?;
        data.push(b'\n');
        self.input.write_all(&data).await?;
        self.input.flush().await?;
        Ok(())
    }
}

impl BridgeReceiver {
    pub async fn read<T: serde::de::DeserializeOwned>(&mut self) -> Result<Option<T>> {
        self.output
            .next_line()
            .await?
            .map(|line| {
                serde_json::from_str(&line)
                    .map_err(|e| AgentError::Model(format!("invalid realtime bridge event: {e}")))
            })
            .transpose()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    pub playback_feedback: bool,
    pub tool_calls: bool,
    pub text_input: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioFormats {
    pub input_rate: u32,
    pub output_rate: u32,
    pub channels: u16,
    pub encoding: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgeInfo {
    pub provider: String,
    pub audio: AudioFormats,
    pub capabilities: Capabilities,
}

impl BridgeInfo {
    pub fn validate(&self) -> Result<()> {
        if self.audio.input_rate != 16000
            || self.audio.output_rate == 0
            || self.audio.channels != 1
            || self.audio.encoding != "pcm_s16le"
        {
            return Err(AgentError::Model(
                "unsupported realtime bridge audio format".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PlaybackFeedback {
    #[serde(rename = "playback_chunk_completed")]
    ChunkCompleted {
        chunk_id: String,
        run_id: String,
        turn_id: String,
    },
    #[serde(rename = "playback_completed")]
    Completed { run_id: String, turn_id: String },
    #[serde(rename = "playback_stopped")]
    Stopped {
        run_id: String,
        turn_id: String,
        played_audio_ms: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn feedback_preserves_provider_identifiers() {
        let feedback = PlaybackFeedback::ChunkCompleted {
            chunk_id: "chunk".into(),
            run_id: "run".into(),
            turn_id: "turn".into(),
        };
        assert_eq!(
            serde_json::to_value(feedback).unwrap(),
            json!({
                "type": "playback_chunk_completed", "chunk_id": "chunk", "run_id": "run", "turn_id": "turn"
            })
        );
    }

    #[test]
    fn capabilities_are_explicit() {
        let value = json!({"provider":"moss", "audio":{
            "input_rate":16000,"output_rate":48000,"channels":1,"encoding":"pcm_s16le"
        }, "capabilities":{"playback_feedback":true,"tool_calls":false,"text_input":false}});
        let info: BridgeInfo = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(info).unwrap(), value);
    }

    #[test]
    fn playback_feedback_is_supported() {
        let value = json!({"provider":"moss", "audio":{
            "input_rate":16000,"output_rate":48000,"channels":1,"encoding":"pcm_s16le"
        }, "capabilities":{"playback_feedback":true,"tool_calls":false,"text_input":false}});
        let info: BridgeInfo = serde_json::from_value(value).unwrap();
        info.validate().unwrap();
    }
}
