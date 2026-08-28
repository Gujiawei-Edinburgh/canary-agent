use canary_agent_revision::{
    AgentBuildRef, AgentId, AgentRevision, AgentSpec, BranchName, BranchRef, CommitMessage,
    ComponentRef, ModelSpec, PromptSpec, RevisionController, RevisionError, RevisionFuture,
    RevisionId, RevisionStore, RuntimePolicySpec, ToolChange, ToolInterfaceSpec, ToolSpec,
    TurnExecutionLimitsSpec,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Mutex;

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

    let base_spec = agent_spec();
    let base = controller
        .commit(
            &main,
            base_spec,
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

    // A revision checkout returns the static agent specification. A future
    // runtime integration can materialize this spec into a live Agent.
    let checked_out = controller.checkout(&update_branch).await?;
    let mut updated_spec = checked_out.revision.spec;
    updated_spec
        .tools
        .insert("compare_results".to_string(), comparison_tool());

    let updated = controller
        .commit(
            &update_branch,
            updated_spec,
            CommitMessage::with_body(
                "add result comparison capability",
                Some("Enable the agent to compare search results explicitly."),
            )?,
        )
        .await?;
    println!("updated revision: {}", updated.revision_id.0);

    // Resolve the branch head, load its immutable revision, and use the
    // stored spec as the updated agent definition.
    let current_revision = controller.checkout(&update_branch).await?.revision;
    let loaded_revision = controller
        .load_revision(&current_revision.revision_id)
        .await?;
    let loaded_spec = loaded_revision.spec;
    println!("checked out branch: {}", update_branch);
    println!("loaded tools: {}", loaded_spec.tools.len());

    let diff = base.diff(&updated)?;
    println!(
        "structural diff:\n{}",
        serde_json::to_string_pretty(&diff).unwrap()
    );
    if diff
        .tools
        .iter()
        .any(|change| matches!(change, ToolChange::Added { name, .. } if name == "compare_results"))
    {
        println!("diff confirms compare_results was added");
    }

    Ok(())
}

fn agent_spec() -> AgentSpec {
    AgentSpec {
        agent_id: AgentId::from("research-agent"),
        model: ModelSpec {
            fqn: "example-provider/example-model".to_string(),
            settings: json!({"reasoning_effort": "medium"}),
        },
        prompts: PromptSpec {
            system: "Research carefully and cite sources.".to_string(),
            templates: BTreeMap::new(),
        },
        tools: BTreeMap::from([("search".to_string(), search_tool())]),
        runtime: RuntimePolicySpec {
            context_builder: ComponentRef::new("example::ContextBuilder", json!({}))
                .expect("context builder"),
            hooks: Vec::new(),
            turn_execution_limits: TurnExecutionLimitsSpec {
                max_model_iterations: 128,
                max_function_calls: 1024,
            },
        },
        build: AgentBuildRef {
            id: "revision-example-build-1".to_string(),
        },
    }
}

fn search_tool() -> ToolSpec {
    ToolSpec {
        interface: ToolInterfaceSpec {
            name: "search".to_string(),
            description: "Search the configured information source.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }),
            output_schema: json!({
                "type": "object",
                "properties": {"results": {"type": "array"}},
                "required": ["results"]
            }),
        },
        configuration: json!({"backend": "example"}),
    }
}

fn comparison_tool() -> ToolSpec {
    ToolSpec {
        interface: ToolInterfaceSpec {
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
            output_schema: json!({
                "type": "object",
                "properties": {"equal": {"type": "boolean"}},
                "required": ["equal"]
            }),
        },
        configuration: json!({"backend": "example"}),
    }
}
