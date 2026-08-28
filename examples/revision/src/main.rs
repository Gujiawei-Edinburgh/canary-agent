use canary_agent_openai::{ChatCompletionsClient, ModelConfig};
use canary_agent_revision::{
    AgentDiff, AgentId, AgentRevision, BranchName, BranchRef, CommitMessage, RevisionController,
    RevisionError, RevisionFuture, RevisionId, RevisionStore, ToolChange, ToolDiff, ValueChange,
};
use canary_agent_runtime::{
    builtin_registry, Agent, AgentConfig, FunctionExecution, FunctionLimits,
    FunctionRecoveryPolicy, FunctionSpec, LocalSessionCoordinator, SimpleFunction,
};
use canary_agent_store_json::JsonFileThreadStore;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use tempfile::tempdir;

#[derive(Default)]
struct InMemRevisionStore {
    revisions: Mutex<BTreeMap<RevisionId, AgentRevision>>,
    branches: Mutex<BTreeMap<BranchRef, RevisionId>>,
}

impl InMemRevisionStore {
    fn new() -> Self {
        Self::default()
    }
}

impl RevisionStore for InMemRevisionStore {
    fn load_revision<'a>(
        &'a self,
        id: &'a RevisionId,
    ) -> RevisionFuture<'a, Option<AgentRevision>> {
        Box::pin(async move {
            let revisions = self
                .revisions
                .lock()
                .map_err(|_| RevisionError::Store("revision lock poisoned".to_string()))?;
            Ok(revisions.get(id).cloned())
        })
    }

    fn save_revision<'a>(&'a self, revision: &'a AgentRevision) -> RevisionFuture<'a, ()> {
        Box::pin(async move {
            let mut revisions = self
                .revisions
                .lock()
                .map_err(|_| RevisionError::Store("revision lock poisoned".to_string()))?;
            if let Some(existing) = revisions.get(&revision.revision_id) {
                if existing != revision {
                    return Err(RevisionError::Store(format!(
                        "revision object {} already contains different data",
                        revision.revision_id.0
                    )));
                }
            }
            revisions.insert(revision.revision_id.clone(), revision.clone());
            Ok(())
        })
    }

    fn branch_head<'a>(&'a self, branch: &'a BranchRef) -> RevisionFuture<'a, Option<RevisionId>> {
        Box::pin(async move {
            let branches = self
                .branches
                .lock()
                .map_err(|_| RevisionError::Store("branch lock poisoned".to_string()))?;
            Ok(branches.get(branch).cloned())
        })
    }

    fn compare_and_set_branch<'a>(
        &'a self,
        branch: &'a BranchRef,
        expected: Option<&'a RevisionId>,
        next: &'a RevisionId,
    ) -> RevisionFuture<'a, bool> {
        Box::pin(async move {
            let mut branches = self
                .branches
                .lock()
                .map_err(|_| RevisionError::Store("branch lock poisoned".to_string()))?;
            if branches.get(branch) != expected {
                return Ok(false);
            }
            branches.insert(branch.clone(), next.clone());
            Ok(true)
        })
    }
}

#[tokio::main]
async fn main() -> canary_agent_revision::Result<()> {
    let store = InMemRevisionStore::new();
    let controller = RevisionController::new(store);
    let main = BranchRef::new(AgentId::from("research-agent"), BranchName::from("main"));

    let state_dir = tempdir().map_err(|error| RevisionError::Store(error.to_string()))?;
    let thread_store = Arc::new(
        JsonFileThreadStore::open(state_dir.path())
            .map_err(|error| RevisionError::Store(error.to_string()))?,
    );
    let model_client = Arc::new(ChatCompletionsClient::new(ModelConfig {
        base_url: "https://example.invalid/v1".to_string(),
        api_key: "example-key".to_string(),
        model: "example-model".to_string(),
        reasoning_effort: "medium".to_string(),
    }));
    let mut registry = builtin_registry();
    registry.register(SimpleFunction::new(
        FunctionSpec {
            name: "search".to_string(),
            description: "Search the configured information source.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }),
        },
        json!({
            "type": "object",
            "properties": {"results": {"type": "array"}},
            "required": ["results"]
        }),
        FunctionLimits {
            time_budget: std::time::Duration::from_secs(5),
            max_output_bytes: 20 * 1024 * 1024,
        },
        FunctionRecoveryPolicy::Idempotent,
        |_args, _context| async { Ok(FunctionExecution::Completed { output: json!({}) }) },
    ));
    registry.register(SimpleFunction::new(
        FunctionSpec {
            name: "legacy_lookup".to_string(),
            description: "Legacy lookup capability kept for the first build.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"]
            }),
        },
        json!({"type": "object"}),
        FunctionLimits {
            time_budget: std::time::Duration::from_secs(5),
            max_output_bytes: 20 * 1024 * 1024,
        },
        FunctionRecoveryPolicy::Idempotent,
        |_args, _context| async { Ok(FunctionExecution::Completed { output: json!({}) }) },
    ));
    let agent = Agent::new(
        AgentConfig {
            agent_id: "research-agent".to_string(),
            system_prompt: "Research carefully and cite sources.".to_string(),
            ..AgentConfig::default()
        },
        thread_store,
        model_client,
        registry,
        Arc::new(LocalSessionCoordinator::default()),
    );

    let snapshot1 = agent.spec_snapshot("build1");
    let base = controller
        .commit(
            &main,
            snapshot1,
            CommitMessage::with_body(
                "create research agent",
                Some("Start with a focused search capability."),
            )?,
        )
        .await?;
    println!("base revision: {}", base.revision_id.0);

    let update_branch = BranchRef::new(
        AgentId::from("research-agent"),
        BranchName::from("add-comparison"),
    );
    controller
        .create_branch(&main, update_branch.clone())
        .await?;

    let updated_agent = agent
        .without_function("legacy_lookup")
        .with_function(SimpleFunction::new(
            FunctionSpec {
                name: "compare_results".to_string(),
                description: "Compare two previously collected result sets.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "left": {"type": "array"},
                        "right": {"type": "array"}
                    },
                    "required": ["left", "right"]
                }),
            },
            json!({
                "type": "object",
                "properties": {"equal": {"type": "boolean"}},
                "required": ["equal"]
            }),
            FunctionLimits {
                time_budget: std::time::Duration::from_secs(5),
                max_output_bytes: 20 * 1024 * 1024,
            },
            FunctionRecoveryPolicy::Idempotent,
            |_args, _context| async {
                Ok(FunctionExecution::Completed {
                    output: json!({"equal": true}),
                })
            },
        ));
    let snapshot2 = updated_agent.spec_snapshot("build2");

    let updated = controller
        .commit(
            &update_branch,
            snapshot2,
            CommitMessage::with_body(
                "add result comparison capability",
                Some("Enable the agent to compare search results explicitly."),
            )?,
        )
        .await?;
    println!("updated revision: {}", updated.revision_id.0);

    let merged = controller
        .merge(
            &update_branch,
            &main,
            CommitMessage::new("merge comparison capability")?,
        )
        .await?;
    println!("merged revision: {}", merged.revision_id.0);

    let diff = controller
        .diff(&base.revision_id, &updated.revision_id)
        .await?;
    println!(
        "{}",
        format_git_diff(&diff, &base.revision_id.0, &updated.revision_id.0)
    );
    if diff
        .tools
        .iter()
        .any(|change| matches!(change, ToolChange::Added { name, .. } if name == "compare_results"))
    {
        println!("diff confirms compare_results was added");
    }
    if diff
        .tools
        .iter()
        .any(|change| matches!(change, ToolChange::Removed { name, .. } if name == "legacy_lookup"))
    {
        println!("diff confirms legacy_lookup was removed");
    }

    Ok(())
}

fn format_git_diff(diff: &AgentDiff, before: &str, after: &str) -> String {
    const GREEN: &str = "\x1b[32m";
    const RED: &str = "\x1b[31m";
    const RESET: &str = "\x1b[0m";
    let mut output = format!("diff --agent\n--- {before}\n+++ {after}\n");
    append_change(&mut output, "agent_id", diff.agent_id.as_ref());
    append_change(&mut output, "model", diff.model.as_ref());
    append_change(&mut output, "prompts", diff.prompts.as_ref());
    append_change(&mut output, "runtime", diff.runtime.as_ref());
    append_change(&mut output, "build", diff.build.as_ref());

    for change in &diff.tools {
        match change {
            ToolChange::Added { name, tool } => {
                output.push_str(&format!("@@ tools.{name} @@\n"));
                append_json(&mut output, "+", GREEN, RESET, tool);
            }
            ToolChange::Removed { name, tool } => {
                output.push_str(&format!("@@ tools.{name} @@\n"));
                append_json(&mut output, "-", RED, RESET, tool);
            }
            ToolChange::Modified { name, diff } => {
                append_tool_diff(&mut output, name, diff);
            }
        }
    }

    output
}

fn append_tool_diff(output: &mut String, name: &str, diff: &ToolDiff) {
    append_change(
        output,
        &format!("tools.{name}.interface"),
        diff.interface.as_ref(),
    );
    append_change(
        output,
        &format!("tools.{name}.configuration"),
        diff.configuration.as_ref(),
    );
}

fn append_change<T: serde::Serialize>(
    output: &mut String,
    path: &str,
    change: Option<&ValueChange<T>>,
) {
    let Some(change) = change else {
        return;
    };
    output.push_str(&format!("@@ {path} @@\n"));
    append_json(output, "-", "\x1b[31m", "\x1b[0m", &change.before);
    append_json(output, "+", "\x1b[32m", "\x1b[0m", &change.after);
}

fn append_json<T: serde::Serialize>(
    output: &mut String,
    prefix: &str,
    color: &str,
    reset: &str,
    value: &T,
) {
    let encoded = serde_json::to_string_pretty(value).expect("diff value is serializable");
    for line in encoded.lines() {
        output.push_str(color);
        output.push_str(prefix);
        output.push_str(line);
        output.push_str(reset);
        output.push('\n');
    }
}
