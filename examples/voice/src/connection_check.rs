use crate::{config::VoiceConfig, qwen::QwenClient, tools};
use canary_agent_kernel::{ModelRequest, Thread};
use canary_agent_runtime::{ModelClient, ModelSessionCommand, ModelSessionConfig, Result};

pub async fn check(config: VoiceConfig) -> Result<()> {
    let functions = tools::registry(config.workspace).specs();
    println!(
        "Checking Qwen Python SDK session: model={}, voice={}, tool schemas={}. No audio or inference.",
        config.model.model,
        config.model.voice,
        functions.len()
    );
    let client = QwenClient::new(config.model);
    let mut session = client
        .connect(ModelSessionConfig {
            connection_id: "connection_check".into(),
            turn_id: "connection_check".into(),
            request: ModelRequest {
                messages: vec![],
                functions,
            },
            history: Thread::new("connection_check"),
            model: client.model_descriptor(),
        })
        .await?;
    println!("session.updated received; session setup succeeded.");
    session.sender.send(ModelSessionCommand::Close).await
}
