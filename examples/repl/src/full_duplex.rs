use canary_agent_kernel::{TurnItemKind, TurnStatus};
use canary_agent_runtime::{
    Agent, AgentError, FullDuplexConfig, FullDuplexEvent, ModelStreamEvent, Result, ThreadStore,
    TurnOutcome,
};
use canary_agent_store_json::JsonFileThreadStore;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use tokio::sync::mpsc;

pub async fn run(agent: Agent, store: Arc<JsonFileThreadStore>, thread_id: String) -> Result<()> {
    println!("thread: {thread_id} (full duplex)");
    println!("Enter messages while the agent is working. /quit exits; Ctrl+C aborts an open turn.");
    // A dedicated OS thread avoids Tokio stdin's uncancellable blocking read
    // keeping runtime shutdown alive after Ctrl+C.
    let (sender, mut input) = mpsc::channel(16);
    std::thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            if sender.blocking_send(line).is_err() {
                break;
            }
        }
    });
    let result = run_interactions(&agent, &store, &thread_id, &mut input).await;
    if let Ok(thread) = store.load(&thread_id).await {
        println!("\nthread token usage: {}", thread.token_usage);
    }
    result
}

async fn run_interactions(
    agent: &Agent,
    store: &JsonFileThreadStore,
    thread_id: &str,
    input: &mut mpsc::Receiver<io::Result<String>>,
) -> Result<()> {
    let mut recovery = store.load(thread_id).await.ok().is_some_and(|thread| {
        thread.turns.last().is_some_and(|turn| {
            turn.status != TurnStatus::Completed
                && turn
                    .items
                    .iter()
                    .any(|item| matches!(item.kind, TurnItemKind::FullDuplexOpened))
        })
    });
    let mut pending_input = None;
    loop {
        let mut turn = if recovery {
            recovery = false;
            agent
                .recover_full_duplex_turn(thread_id, FullDuplexConfig::default())
                .await?
        } else {
            print!("> ");
            io::stdout().flush()?;
            let line = if let Some(line) = pending_input.take() {
                line
            } else {
                tokio::select! {
                    signal = tokio::signal::ctrl_c() => { signal?; return Ok(()); }
                    line = input.recv() => match line { Some(line) => line?, None => return Ok(()) },
                }
            };
            if matches!(line.trim(), "/quit" | "/exit") {
                return Ok(());
            }
            if line.trim().is_empty() {
                continue;
            }
            agent
                .open_full_duplex_turn(
                    thread_id,
                    line,
                    serde_json::json!({}),
                    FullDuplexConfig::default(),
                )
                .await?
        };
        let handle = turn.handle();
        let mut exit = false;
        let mut streamed = false;
        loop {
            tokio::select! {
                signal = tokio::signal::ctrl_c() => { signal?; handle.abort(); break; }
                line = input.recv() => match line {
                    Some(Ok(line)) if matches!(line.trim(), "/quit" | "/exit") => { exit = true; handle.abort(); break; }
                    Some(Ok(line)) if !line.trim().is_empty() => {
                        if handle.append_input(line.clone()).await.is_err() {
                            pending_input = Some(line);
                            break;
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(error)) => return Err(AgentError::Io(error)),
                    None => { exit = true; handle.abort(); break; }
                },
                event = turn.next_event() => match event {
                    Ok(FullDuplexEvent::Model { event: ModelStreamEvent::AssistantDelta { text }, .. }) => {
                        streamed = true; print!("{text}"); io::stdout().flush()?;
                    }
                    Ok(FullDuplexEvent::ItemCommitted { item }) => match item.kind {
                        TurnItemKind::InferenceStarted { .. } => streamed = false,
                        TurnItemKind::InferenceInterrupted { .. } => println!("\n[response interrupted]"),
                        TurnItemKind::ModelResponse { text: Some(text), .. } if !streamed => println!("{text}"),
                        TurnItemKind::ToolAccepted { name, operation_id, .. } => println!("\n[accepted] {name}: {operation_id}"),
                        TurnItemKind::ToolCompleted { name, result, .. } => println!("\n[completed] {name}: {}", serde_json::to_string(&result)?),
                        TurnItemKind::ToolSuspended { operation_id, .. } => println!("\n[suspended] {operation_id}; resolve through the host API"),
                        TurnItemKind::ToolIndeterminate { operation_id, reason } => println!("\n[indeterminate] {operation_id}: {reason}"),
                        _ => {}
                    },
                    Ok(FullDuplexEvent::Idle) => { print!("\n> "); io::stdout().flush()?; }
                    Ok(FullDuplexEvent::Finished { .. }) | Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(count)) => println!("\n[missed {count} events; durable history is in the thread store]"),
                    _ => {}
                }
            }
        }
        match turn.wait().await? {
            TurnOutcome::Failed { error } => println!("\n[failed] {error}"),
            TurnOutcome::Aborted { reason } => println!("\n[aborted] {reason}"),
            _ => println!(),
        }
        if exit {
            return Ok(());
        }
        // An indeterminate operation must be reconciled before new work starts.
        if store.load(thread_id).await.is_ok_and(|thread| {
            !canary_agent_kernel::ThreadProjection::from_thread(&thread)
                .active_operations
                .is_empty()
        }) {
            return Err(AgentError::FullDuplex(
                "unfinished operations require recovery and host reconciliation".into(),
            ));
        }
    }
}
