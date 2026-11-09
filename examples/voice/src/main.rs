mod audio;
mod bridge;
mod config;
mod connection_check;
mod delegation;
mod feedback;
mod model_bridge;
mod playback;
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
use model_bridge::BridgeClient;
use std::{
    io::{self, BufRead, Write},
    sync::Arc,
};
use tokio::sync::{mpsc, watch};

const SYSTEM_PROMPT: &str = include_str!("../prompts/voice.md");

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "usage: cargo run -p voice --bin voice -- [--p qwen|moss] [--check-config | --check-connection]\nLoads examples/voice/.env. Continuous microphone input; /mute, /unmute, /stop, /reconnect, /quit. Ctrl+C exits."
        );
        return Ok(());
    }
    let mut provider = "qwen";
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--p" => {
                index += 1;
                provider = args
                    .get(index)
                    .ok_or_else(|| AgentError::Model("--p requires qwen or moss".into()))?;
                if !matches!(provider, "qwen" | "moss") {
                    return Err(AgentError::Model("--p requires qwen or moss".into()));
                }
            }
            "--check-config" | "--check-connection" => {}
            _ => return Err(AgentError::Model("unknown argument; use --help".into())),
        }
        index += 1;
    }
    let config = VoiceConfig::load(provider)?;
    let playback_control = audio::PlaybackControl::default();
    let mut registry = delegation::registry(config.workspace.clone())?;
    playback::register_tools(&mut registry, playback_control.clone());
    let functions = registry.specs();
    let function_names: Vec<_> = functions
        .iter()
        .map(|function| function.name.clone())
        .collect();
    if args.iter().any(|arg| arg == "--check-connection") {
        return connection_check::check(config, functions).await;
    }
    if args.iter().any(|arg| arg == "--check-config") {
        println!(
            "configuration valid: provider={provider}, model={}, voice={}, input={} Hz",
            config.model.model,
            config.model.voice,
            config.model.input_format().sample_rate
        );
        println!("configured tools: {}", function_names.join(", "));
        return Ok(());
    }
    let _logging =
        init_file_logging(&config.state_dir).map_err(|e| AgentError::Logging(e.to_string()))?;
    let store = Arc::new(JsonFileThreadStore::open(&config.state_dir)?);
    let trace = Arc::new(JsonlTraceCollector::new(&config.state_dir)?);
    let input_format = config.model.input_format();
    let (model, playback_events) = BridgeClient::new(config.model).with_playback();
    let agent = Agent::new(
        AgentConfig {
            system_prompt: SYSTEM_PROMPT.into(),
            ..AgentConfig::default()
        },
        store.clone(),
        Arc::new(model),
        registry,
        Arc::new(LocalSessionCoordinator::default()),
    )
    .with_shared_trace_collector(trace.clone());
    println!("thread: {}", config.thread_id);
    println!("configured tools: {}", function_names.join(", "));
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
                serde_json::json!({"application":"voice", "configured_tools":function_names}),
                TurnConfig::default(),
            )
            .await?
    };
    let result = interact(turn, input_format, playback_events, playback_control).await;
    trace.flush().await;
    if let Ok(thread) = store.load(&config.thread_id).await {
        println!("\nthread token usage: {}", thread.token_usage);
    }
    result
}

async fn interact(
    mut turn: RunningTurn,
    format: canary_agent_runtime::AudioFormat,
    mut playback_events: feedback::Events,
    playback_control: audio::PlaybackControl,
) -> Result<()> {
    let handle = turn.handle();
    let result = async {
        let mut audio = AudioDevices::open(format, playback_control)?;
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
        let mut text_input = false;
        let mut playback_events_open = true;
        println!("Connecting. Echo cancellation is active. Speak after [listening].");
        loop {
            send_feedback(&audio).await;
            let deadline = audio.feedback_deadline();
            let feedback_changed = audio.feedback_changed.clone();
            tokio::select! {
                // Drain already-received media before logical completion can
                // close the interaction, but always give user abort priority.
                biased;
                signal = tokio::signal::ctrl_c() => {
                    signal?;
                    audio.interrupt(); send_feedback(&audio).await;
                    handle.abort();
                    break;
                }
                _ = feedback_changed.notified() => {},
                _ = async {
                    match deadline {
                        Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                        None => std::future::pending().await,
                    }
                } => {},
                event = playback_events.recv(), if playback_events_open => match event {
                    Some(feedback::Event::Ready { connection_id, provider, tool_calls, text_input: supported }) => {
                        audio.connect(&connection_id); text_input = supported;
                        println!("[playback connected] {connection_id}");
                        println!("[bridge capabilities] provider={provider}, custom_tools={tool_calls}, text_input={supported}");
                        if !tool_calls {
                            eprintln!("[warning] This bridge does not register Canary tools. Project files, exec_command, web_search, and delegate are unavailable. Any tools mentioned in speech are not evidence of Canary tool access.");
                        }
                    }
                    Some(feedback::Event::Audio { response_id, chunk_id, chunk, target }) => {
                        if !chunk.data.is_empty() {
                            audio.enqueue(&response_id, chunk, target.map(|target| audio::Tracking { chunk_id, target }))?;
                        }
                    }
                    Some(feedback::Event::Stop { response_id, target }) => { audio.stop_target(&response_id, &target); }
                    Some(feedback::Event::TextDelta { response_id, text }) => { audio.text_delta(&response_id, &text); }
                    Some(feedback::Event::AudioDone { target }) => { audio.audio_done(target)?; }
                    Some(feedback::Event::Finished { response_id, .. }) => {
                        audio.finish_response(&response_id)?;
                    }
                    None => playback_events_open = false,
                },
                changed = audio.errors.changed() => {
                    if changed.is_ok() && let Some(error) = audio.errors.borrow().clone() {
                        return Err(AgentError::Model(error));
                    }
                }
                line = lines.recv(), if stdin_open => {
                    match line {
                        Some(Ok(line)) => match line.trim() {
                            "/quit" => { audio.interrupt(); send_feedback(&audio).await; handle.abort(); break; }
                            "/stop" => { audio.interrupt(); send_feedback(&audio).await; println!("\n[playback stopped]"); }
                            "/mute" => { muted = true; let _ = enabled.send(if ready { MicrophoneMode::Muted } else { MicrophoneMode::Off }); println!("\n[muted]"); }
                            "/unmute" => { muted = false; let _ = enabled.send(if ready { MicrophoneMode::Live } else { MicrophoneMode::Off }); println!("\n[listening]"); }
                            "/reconnect" => match handle.reconnect_model().await { Ok(()) => println!("\n[reconnecting]"), Err(error) => eprintln!("{error}") },
                            "" => {},
                            text => if !text_input {
                                eprintln!("This bridge does not support typed input.");
                            } else if let Err(error) = handle.append_input(text).await { eprintln!("{error}"); },
                        },
                        Some(Err(error)) => return Err(error.into()),
                        None => stdin_open = false,
                    }
                }
                event = turn.next_event() => match event {
                    Ok(canary_agent_runtime::TurnEvent::SessionReady { .. }) => { ready = true; let _ = enabled.send(if muted { MicrophoneMode::Muted } else { MicrophoneMode::Live }); println!("[listening]"); }
                    Ok(canary_agent_runtime::TurnEvent::Session { event, .. }) => match event {
                        ModelSessionEvent::InputStarted { .. } => { println!("\n[user speaking]"); }
                        ModelSessionEvent::TranscriptFinalized { text, .. } => println!("\nuser: {text}"),
                        ModelSessionEvent::TextDelta { text, .. } => { print!("{text}"); io::stdout().flush()?; }
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
                        send_feedback(&audio).await;
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

async fn send_feedback(audio: &AudioDevices) {
    for delivery in audio.feedback() {
        if let Err(error) = delivery.send().await {
            // A disconnected session cannot receive acknowledgments. Never
            // redirect these to a replacement connection.
            eprintln!("[playback feedback unavailable] {error}");
        }
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
