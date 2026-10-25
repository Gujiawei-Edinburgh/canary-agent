mod audio;
mod config;
mod connection_check;
mod qwen;
mod tools;

use audio::AudioDevices;
use canary_agent_kernel::{ThreadProjection, TurnItemKind, TurnStatus};
use canary_agent_observability::{JsonlTraceCollector, init_file_logging};
use canary_agent_runtime::{
    Agent, AgentConfig, AgentError, LocalSessionCoordinator, ModelInput, ModelSessionEvent, Result,
    RunningTurn, ThreadStore, TraceCollector, TurnConfig,
};
use canary_agent_store_json::JsonFileThreadStore;
use config::VoiceConfig;
use qwen::QwenClient;
use std::{
    io::{self, BufRead, Write},
    sync::Arc,
};
use tokio::sync::{mpsc, watch};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "usage: cargo run -p voice -- [--check-config | --check-connection]\nLoads examples/voice/.env. Continuous microphone input; /mute, /unmute, /reconnect, /quit. Ctrl+C exits."
        );
        return Ok(());
    }
    if args.len() > 1
        || args
            .iter()
            .any(|arg| arg != "--check-config" && arg != "--check-connection")
    {
        return Err(AgentError::Model("unknown argument; use --help".into()));
    }
    let config = VoiceConfig::load()?;
    if args.iter().any(|arg| arg == "--check-connection") {
        return connection_check::check(config).await;
    }
    if args.iter().any(|arg| arg == "--check-config") {
        println!(
            "configuration valid: Qwen Python SDK; model={}, voice={}, input={} Hz, output=24000 Hz",
            config.model.model,
            config.model.voice,
            config.model.input_format().sample_rate
        );
        return Ok(());
    }
    let _logging =
        init_file_logging(&config.state_dir).map_err(|e| AgentError::Logging(e.to_string()))?;
    let store = Arc::new(JsonFileThreadStore::open(&config.state_dir)?);
    let trace = Arc::new(JsonlTraceCollector::new(&config.state_dir)?);
    let input_format = config.model.input_format();
    let agent = Agent::new(
        AgentConfig {
        system_prompt:concat!("You are a conversational voice assistant. Respond naturally in the user's language. ",
            "Keep listening while tools work. A single answer is not the end of the interaction. ",
            "Call end_turn only when the user explicitly ends the conversation. ",
            "Use tools for current facts and project inspection; do not invent tool results. ",
            "If the user changes their mind, decide whether existing work should continue or use cancel_operation.").into(),
        ..AgentConfig::default()
        },
        store.clone(),
        Arc::new(QwenClient::new(config.model)),
        tools::registry(config.workspace),
        Arc::new(LocalSessionCoordinator::default())
    ).with_shared_trace_collector(trace.clone());
    println!("thread: {}", config.thread_id);
    let recovering = store
        .load(&config.thread_id)
        .await
        .ok()
        .is_some_and(|thread| {
            thread.turns.last().is_some_and(|turn| {
                matches!(turn.status, TurnStatus::Running | TurnStatus::Suspended)
            }) || !ThreadProjection::from_thread(&thread)
                .active_operations
                .is_empty()
        });
    let turn = if recovering {
        agent
            .recover_turn(&config.thread_id, TurnConfig::default())
            .await?
    } else {
        agent
            .open_stream_turn(
                &config.thread_id,
                serde_json::json!({"application":"voice"}),
                TurnConfig::default(),
            )
            .await?
    };
    let result = interact(turn, input_format).await;
    trace.flush().await;
    if let Ok(thread) = store.load(&config.thread_id).await {
        println!("\nthread token usage: {}", thread.token_usage);
    }
    result
}

async fn interact(mut turn: RunningTurn, format: canary_agent_runtime::AudioFormat) -> Result<()> {
    let handle = turn.handle();
    let result = async {
        let mut audio = AudioDevices::open(format)?;
        let (enabled, input_enabled) = watch::channel(MicrophoneMode::Off);
        let _microphone = forward_microphone(
            audio.microphone.take().expect("microphone receiver"),
            handle.clone(),
            input_enabled
        );
        let (commands,mut lines) = mpsc::channel(16);
        std::thread::spawn(move || {
            for line in io::stdin().lock().lines() {
                if commands.blocking_send(line).is_err() { break; }
            }
        });
        let mut ready = false;
        let mut muted = false;
        let mut stdin_open = true;
        println!("Connecting. Echo cancellation is active. Speak after [listening].");
        loop {
            tokio::select! {
                signal = tokio::signal::ctrl_c() => {
                    signal?;
                    handle.abort();
                    break;
                }
                changed = audio.errors.changed() => {
                    if changed.is_ok() && let Some(error) = audio.errors.borrow().clone() {
                        return Err(AgentError::Model(error));
                    }
                }
                line = lines.recv(), if stdin_open => {
                    match line {
                        Some(Ok(line)) => match line.trim() {
                            "/quit" => { handle.abort(); break; }
                            "/mute" => { muted = true; let _ = enabled.send(if ready { MicrophoneMode::Muted } else { MicrophoneMode::Off }); println!("\n[muted]"); }
                            "/unmute" => { muted = false; let _ = enabled.send(if ready { MicrophoneMode::Live } else { MicrophoneMode::Off }); println!("\n[listening]"); }
                            "/reconnect" => match handle.reconnect_model().await { Ok(()) => println!("\n[reconnecting]"), Err(error) => eprintln!("{error}") },
                            "" => {},
                            text => if let Err(error) = handle.append_input(text).await { eprintln!("{error}"); },
                        },
                        Some(Err(error)) => return Err(error.into()),
                        None => stdin_open = false,
                    }
                }
                event = turn.next_event() => match event {
                    Ok(canary_agent_runtime::TurnEvent::SessionReady { .. }) => { ready = true; let _ = enabled.send(if muted { MicrophoneMode::Muted } else { MicrophoneMode::Live }); println!("[listening]"); }
                    Ok(canary_agent_runtime::TurnEvent::Session { event, .. }) => match event {
                        ModelSessionEvent::AudioDelta { chunk, .. } => audio.enqueue(chunk)?,
                        ModelSessionEvent::InputStarted { .. } => { audio.interrupt(); println!("\n[user speaking]"); }
                        ModelSessionEvent::TranscriptFinalized { text, .. } => println!("\nuser: {text}"),
                        ModelSessionEvent::TextDelta { text, .. } => { print!("{text}"); io::stdout().flush()?; }
                        ModelSessionEvent::ResponseFinished { status:canary_agent_runtime::ModelResponseStatus::Completed, .. } => audio.finish_response()?,
                        ModelSessionEvent::ResponseFinished { status:canary_agent_runtime::ModelResponseStatus::Interrupted { .. }, .. } => audio.interrupt(),
                        _ => {},
                    },
                    Ok(canary_agent_runtime::TurnEvent::Suspended { suspension }) => { ready = false; let _ = enabled.send(MicrophoneMode::Off); audio.interrupt(); println!("\n[connection suspended] {}\nUse /reconnect or Ctrl+C.",suspension.payload); }
                    Ok(canary_agent_runtime::TurnEvent::ItemCommitted { item }) => match item.kind {
                        TurnItemKind::ToolAccepted { name,operation_id,.. } => println!("\n[tool] {name}: {operation_id}"),
                        TurnItemKind::ToolCompleted { name,.. } => println!("\n[tool completed] {name}"),
                        TurnItemKind::ToolSuspended { operation_id,.. } | TurnItemKind::ToolIndeterminate { operation_id,.. } => println!("\n[tool requires host resolution] {operation_id}"),
                        _ => {},
                    },
                    Ok(canary_agent_runtime::TurnEvent::Finished { .. }) => {
                        // Playback remains an application concern after logical completion.
                        let _ = enabled.send(MicrophoneMode::Off);
                        tokio::select! {
                            _ = audio.wait_drained() => {},
                            signal = tokio::signal::ctrl_c() => { signal?; audio.interrupt(); },
                        }
                        break;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(error) => return Err(AgentError::Model(format!("voice output stream lost events: {error}"))),
                    _ => {},
                }
            }
        }
        Ok(())
    }.await;
    if result.is_err() {
        handle.abort();
    }
    let outcome = turn.wait().await?;
    println!("\n{outcome:?}");
    result?;
    match outcome {
        canary_agent_runtime::TurnOutcome::Failed { error } => Err(AgentError::Model(error)),
        _ => Ok(()),
    }
}

struct MicrophoneForwarder(tokio::task::JoinHandle<()>);
#[derive(Clone, Copy)]
enum MicrophoneMode {
    Off,
    Live,
    Muted,
}
impl Drop for MicrophoneForwarder {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn forward_microphone(
    mut microphone: mpsc::Receiver<canary_agent_runtime::AudioChunk>,
    handle: canary_agent_runtime::TurnHandle,
    mut enabled: watch::Receiver<MicrophoneMode>,
) -> MicrophoneForwarder {
    MicrophoneForwarder(tokio::spawn(async move {
        while let Some(mut chunk) = microphone.recv().await {
            match *enabled.borrow_and_update() {
                MicrophoneMode::Off => continue,
                MicrophoneMode::Muted => chunk.data.fill(0),
                MicrophoneMode::Live => {}
            }
            tokio::select! {
                _ = enabled.changed() => {},
                _ = handle.send_input(ModelInput::Audio { chunk }) => {},
            }
        }
    }))
}
