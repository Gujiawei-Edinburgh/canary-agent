use canary_agent_revision::{
    AgentBuildRef, AgentId, AgentRevision, AgentSpec, BranchName, BranchRef, CommitMessage,
    ComponentRef, LocalRevisionStore, ModelSpec, PromptSpec, RevisionController, RevisionError,
    RevisionMetadata, RuntimePolicySpec, ToolChange, ToolInterfaceSpec, ToolSpec,
    TurnExecutionLimitsSpec,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn tool(name: &str) -> ToolSpec {
    ToolSpec {
        interface: ToolInterfaceSpec {
            name: name.to_string(),
            description: format!("{name} tool"),
            parameters: json!({"type": "object"}),
            output_schema: json!({"type": "object"}),
        },
        configuration: json!({}),
    }
}

fn spec() -> AgentSpec {
    AgentSpec {
        agent_id: AgentId::from("research-agent"),
        model: ModelSpec {
            fqn: "provider/model-v1".to_string(),
            settings: json!({"reasoning_effort": "medium"}),
        },
        prompts: PromptSpec {
            system: "Answer carefully.".to_string(),
            templates: BTreeMap::new(),
        },
        tools: BTreeMap::from([(String::from("search"), tool("search"))]),
        runtime: RuntimePolicySpec {
            context_builder: ComponentRef::new("test::ContextBuilder", json!({}))
                .expect("context builder"),
            hooks: Vec::new(),
            turn_execution_limits: TurnExecutionLimitsSpec {
                max_model_iterations: 128,
                max_function_calls: 1024,
            },
        },
        build: AgentBuildRef {
            id: "application-build-1".to_string(),
        },
    }
}

fn repository_path() -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "canary-agent-revision-{}-{timestamp}",
        std::process::id()
    ))
}

#[tokio::test]
async fn commit_branch_checkout_and_lineage_are_persistent() {
    let path = repository_path();
    let store = LocalRevisionStore::open(&path).await.expect("store");
    let controller = RevisionController::new(store);
    let main = BranchRef::new(AgentId::from("research-agent"), BranchName::from("main"));

    let first = controller
        .commit(
            &main,
            spec(),
            CommitMessage::new("create research agent").expect("message"),
        )
        .await
        .expect("first commit");
    assert!(first.parents.is_empty());

    let feature = BranchRef::new(
        AgentId::from("research-agent"),
        BranchName::from("add-search-v2"),
    );
    let branch_head = controller
        .create_branch(&main, feature.clone())
        .await
        .expect("branch");
    assert_eq!(branch_head, first.revision_id);

    let mut next_spec = spec();
    next_spec.build.id = "application-build-2".to_string();
    let second = controller
        .commit(
            &feature,
            next_spec,
            CommitMessage::new("upgrade search implementation").expect("message"),
        )
        .await
        .expect("second commit");
    assert_eq!(second.parents, vec![first.revision_id.clone()]);
    assert_eq!(
        controller
            .checkout(&feature)
            .await
            .expect("checkout")
            .revision,
        second
    );
    assert_eq!(
        controller
            .parents(&second.revision_id)
            .await
            .expect("parents")[0],
        first
    );
    let history = controller.log(&feature).await.expect("log");
    assert_eq!(
        history
            .iter()
            .map(|revision| revision.message.subject.as_str())
            .collect::<Vec<_>>(),
        vec!["upgrade search implementation", "create research agent"]
    );

    let reopened = RevisionController::new(LocalRevisionStore::open(&path).await.expect("reopen"));
    assert_eq!(
        reopened
            .checkout(&feature)
            .await
            .expect("reopen checkout")
            .revision,
        second
    );
    fs::remove_dir_all(path).expect("cleanup");
}

#[tokio::test]
async fn commit_rejects_a_branch_for_another_agent() {
    let path = repository_path();
    let controller = RevisionController::new(LocalRevisionStore::open(&path).await.expect("store"));
    let branch = BranchRef::new(AgentId::from("other-agent"), BranchName::from("main"));
    let error = controller
        .commit(
            &branch,
            spec(),
            CommitMessage::new("invalid ownership").expect("message"),
        )
        .await
        .expect_err("ownership error");
    assert!(matches!(error, RevisionError::InvalidSpec(_)));
    fs::remove_dir_all(path).expect("cleanup");
}

#[tokio::test]
async fn structural_diff_identifies_tool_subcomponents() {
    let before = AgentRevision::commit(
        spec(),
        Vec::new(),
        CommitMessage::new("baseline").expect("message"),
        RevisionMetadata {
            author: None,
            created_at: None,
        },
    )
    .expect("revision");
    let mut next_spec = spec();
    next_spec
        .tools
        .get_mut("search")
        .expect("search")
        .interface
        .description = "search with filters".to_string();
    let after = AgentRevision::commit(
        next_spec,
        vec![before.revision_id.clone()],
        CommitMessage::new("describe search filters").expect("message"),
        RevisionMetadata {
            author: None,
            created_at: None,
        },
    )
    .expect("revision");

    let diff = before.diff(&after).expect("diff");
    assert!(diff.tools.iter().any(|change| {
        matches!(change, ToolChange::Modified { name, diff }
            if name == "search" && diff.interface.is_some())
    }));
}

#[tokio::test]
async fn merge_combines_independent_static_changes_and_records_two_parents() {
    let path = repository_path();
    let controller = RevisionController::new(LocalRevisionStore::open(&path).await.expect("store"));
    let main = BranchRef::new(AgentId::from("research-agent"), BranchName::from("main"));
    let c0 = BranchRef::new(AgentId::from("research-agent"), BranchName::from("c0"));
    let c1 = BranchRef::new(AgentId::from("research-agent"), BranchName::from("c1"));

    let base = controller
        .commit(
            &main,
            spec(),
            CommitMessage::new("create base agent").expect("message"),
        )
        .await
        .expect("base");
    controller
        .create_branch(&main, c0.clone())
        .await
        .expect("c0 branch");
    controller
        .create_branch(&main, c1.clone())
        .await
        .expect("c1 branch");

    let mut c0_spec = spec();
    c0_spec.prompts.system = "Focus on category c0.".to_string();
    let c0_revision = controller
        .commit(
            &c0,
            c0_spec,
            CommitMessage::new("optimize c0 prompt").expect("message"),
        )
        .await
        .expect("c0 commit");

    let mut c1_spec = spec();
    c1_spec.build.id = "application-build-2".to_string();
    let c1_revision = controller
        .commit(
            &c1,
            c1_spec,
            CommitMessage::new("upgrade search for c1").expect("message"),
        )
        .await
        .expect("c1 commit");

    let merged = controller
        .merge(
            &c1,
            &c0,
            CommitMessage::new("merge c1 search improvements into c0").expect("message"),
        )
        .await
        .expect("merge");
    assert_eq!(
        merged.parents,
        vec![c0_revision.revision_id, c1_revision.revision_id]
    );
    assert_eq!(merged.spec.prompts.system, "Focus on category c0.");
    assert_eq!(merged.spec.build.id, "application-build-2");
    assert_eq!(
        controller
            .checkout(&c0)
            .await
            .expect("merged checkout")
            .revision,
        merged
    );
    assert_eq!(base.parents.len(), 0);
    fs::remove_dir_all(path).expect("cleanup");
}

#[tokio::test]
async fn merge_reports_conflicting_static_changes_without_advancing_branch() {
    let path = repository_path();
    let controller = RevisionController::new(LocalRevisionStore::open(&path).await.expect("store"));
    let main = BranchRef::new(AgentId::from("research-agent"), BranchName::from("main"));
    let left = BranchRef::new(AgentId::from("research-agent"), BranchName::from("left"));
    let right = BranchRef::new(AgentId::from("research-agent"), BranchName::from("right"));
    controller
        .commit(
            &main,
            spec(),
            CommitMessage::new("create base agent").expect("message"),
        )
        .await
        .expect("base");
    controller
        .create_branch(&main, left.clone())
        .await
        .expect("left branch");
    controller
        .create_branch(&main, right.clone())
        .await
        .expect("right branch");

    let mut left_spec = spec();
    left_spec.prompts.system = "left prompt".to_string();
    controller
        .commit(
            &left,
            left_spec,
            CommitMessage::new("change prompt left").expect("message"),
        )
        .await
        .expect("left commit");

    let mut right_spec = spec();
    right_spec.prompts.system = "right prompt".to_string();
    controller
        .commit(
            &right,
            right_spec,
            CommitMessage::new("change prompt right").expect("message"),
        )
        .await
        .expect("right commit");

    let error = controller
        .merge(
            &right,
            &left,
            CommitMessage::new("merge conflicting prompts").expect("message"),
        )
        .await
        .expect_err("conflict");
    assert!(matches!(error, RevisionError::MergeConflicts(_)));
    assert_eq!(
        controller
            .checkout(&left)
            .await
            .expect("checkout")
            .revision
            .message
            .subject,
        "change prompt left"
    );
    fs::remove_dir_all(path).expect("cleanup");
}
