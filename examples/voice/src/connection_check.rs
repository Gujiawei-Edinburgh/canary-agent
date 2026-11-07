use crate::{config::VoiceConfig, feedback, model_bridge::BridgeClient, tools};
use canary_agent_kernel::{ModelRequest, Thread};
use canary_agent_runtime::{ModelClient, ModelSessionCommand, ModelSessionConfig, Result};

pub async fn check(config: VoiceConfig) -> Result<()> {
    let functions = tools::registry(config.workspace).specs();
    println!(
        "Checking realtime bridge session: model={}, voice={}, tool schemas={}. No audio or inference.",
        config.model.model,
        config.model.voice,
        functions.len()
    );
    let (client, mut events) = BridgeClient::new(config.model).with_playback();
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
    if let Some(feedback::Event::Ready {
        provider,
        tool_calls,
        text_input,
        ..
    }) = events.recv().await
    {
        println!(
            "bridge capabilities: provider={provider}, custom_tools={tool_calls}, text_input={text_input}"
        );
        if !tool_calls {
            eprintln!(
                "Custom tools were NOT registered. This check verified session setup, not tool execution."
            );
        }
    }
    session.sender.send(ModelSessionCommand::Close).await
}
