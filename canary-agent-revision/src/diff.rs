use crate::ids::AgentId;
use crate::spec::{
    AgentBuildRef, AgentSpec, ModelSpec, PromptSpec, RuntimePolicySpec, ToolInterfaceSpec, ToolSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentDiff {
    pub agent_id: Option<ValueChange<AgentId>>,
    pub model: Option<ValueChange<ModelSpec>>,
    pub prompts: Option<ValueChange<PromptSpec>>,
    pub tools: Vec<ToolChange>,
    pub runtime: Option<ValueChange<RuntimePolicySpec>>,
    pub build: Option<ValueChange<Option<AgentBuildRef>>>,
    pub extensions: Option<ValueChange<BTreeMap<String, Value>>>,
}

impl AgentDiff {
    pub(crate) fn between(before: &AgentSpec, after: &AgentSpec) -> crate::Result<Self> {
        Ok(Self {
            agent_id: changed(&before.agent_id, &after.agent_id),
            model: changed(&before.model, &after.model),
            prompts: changed(&before.prompts, &after.prompts),
            tools: diff_tools(&before.tools, &after.tools),
            runtime: changed(&before.runtime, &after.runtime),
            build: changed(&before.build, &after.build),
            extensions: changed(&before.extensions, &after.extensions),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.agent_id.is_none()
            && self.model.is_none()
            && self.prompts.is_none()
            && self.tools.is_empty()
            && self.runtime.is_none()
            && self.build.is_none()
            && self.extensions.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueChange<T> {
    pub before: T,
    pub after: T,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolChange {
    Added { name: String, tool: Box<ToolSpec> },
    Removed { name: String, tool: Box<ToolSpec> },
    Modified { name: String, diff: Box<ToolDiff> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDiff {
    pub interface: Option<ValueChange<ToolInterfaceSpec>>,
    pub configuration: Option<ValueChange<Value>>,
    pub extensions: Option<ValueChange<BTreeMap<String, Value>>>,
}

impl ToolDiff {
    pub fn is_empty(&self) -> bool {
        self.interface.is_none() && self.configuration.is_none() && self.extensions.is_none()
    }
}

fn changed<T: PartialEq + Clone>(before: &T, after: &T) -> Option<ValueChange<T>> {
    (before != after).then(|| ValueChange {
        before: before.clone(),
        after: after.clone(),
    })
}

fn diff_tools(
    before: &BTreeMap<String, ToolSpec>,
    after: &BTreeMap<String, ToolSpec>,
) -> Vec<ToolChange> {
    let names = before
        .keys()
        .chain(after.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    names
        .into_iter()
        .filter_map(|name| match (before.get(&name), after.get(&name)) {
            (None, Some(tool)) => Some(ToolChange::Added {
                name,
                tool: Box::new(tool.clone()),
            }),
            (Some(tool), None) => Some(ToolChange::Removed {
                name,
                tool: Box::new(tool.clone()),
            }),
            (Some(before), Some(after)) if before != after => Some(ToolChange::Modified {
                name,
                diff: Box::new(ToolDiff {
                    interface: changed(&before.interface, &after.interface),
                    configuration: changed(&before.configuration, &after.configuration),
                    extensions: changed(&before.extensions, &after.extensions),
                }),
            }),
            _ => None,
        })
        .collect()
}
