use crate::diff::AgentDiff;
use crate::error::{Result, RevisionError};
use crate::ids::{validate_component, AgentId, SpecDigest};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentRef {
    pub fqn: String,
    pub configuration: Value,
}

impl ComponentRef {
    pub fn new(fqn: impl Into<String>, configuration: Value) -> Result<Self> {
        let component = Self {
            fqn: fqn.into(),
            configuration,
        };
        component.validate()?;
        Ok(component)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.fqn.trim().is_empty() {
            return Err(RevisionError::InvalidSpec("component FQN is empty".to_string()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub fqn: String,
    pub settings: Value,
}

impl ModelSpec {
    fn validate(&self) -> Result<()> {
        if self.fqn.trim().is_empty() {
            return Err(RevisionError::InvalidSpec(
                "model provider and model are required".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptSpec {
    pub system: String,
    pub templates: BTreeMap<String, String>,
}

impl PromptSpec {
    fn validate(&self) -> Result<()> {
        for name in self.templates.keys() {
            if name.trim().is_empty() {
                return Err(RevisionError::InvalidSpec(
                    "prompt extension name is empty".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInterfaceSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub output_schema: Value,
}

/// Application-defined identity of the build that supplies the agent runtime
/// and its opaque tool implementations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentBuildRef {
    pub id: String,
}

impl AgentBuildRef {
    fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(RevisionError::InvalidSpec(
                "agent build id is empty".to_string(),
            ));
        }
        Ok(())
    }
}

impl ToolInterfaceSpec {
    fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(RevisionError::InvalidSpec("tool name is empty".to_string()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub interface: ToolInterfaceSpec,
    pub configuration: Value,
}

impl ToolSpec {
    fn validate(&self, expected_name: &str) -> Result<()> {
        self.interface.validate()?;
        if self.interface.name != expected_name {
            return Err(RevisionError::InvalidSpec(format!(
                "tool map key {expected_name} does not match interface name {}",
                self.interface.name
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimePolicySpec {
    pub context_builder: ComponentRef,
    pub hooks: Vec<ComponentRef>,
    pub turn_execution_limits: TurnExecutionLimitsSpec,
}

impl RuntimePolicySpec {
    fn validate(&self) -> Result<()> {
        self.context_builder.validate()?;
        for hook in &self.hooks {
            hook.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnExecutionLimitsSpec {
    pub max_model_iterations: usize,
    pub max_function_calls: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentSpec {
    pub agent_id: AgentId,
    pub model: ModelSpec,
    pub prompts: PromptSpec,
    pub tools: BTreeMap<String, ToolSpec>,
    pub runtime: RuntimePolicySpec,
    pub build: AgentBuildRef,
}

impl AgentSpec {
    pub fn validate(&self) -> Result<()> {
        validate_component("agent id", &self.agent_id.0)
            .map_err(|error| RevisionError::InvalidSpec(error.to_string()))?;
        self.model.validate()?;
        self.prompts.validate()?;
        for (name, tool) in &self.tools {
            tool.validate(name)?;
        }
        self.runtime.validate()?;
        self.build.validate()?;
        Ok(())
    }

    pub fn digest(&self) -> Result<SpecDigest> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| RevisionError::Serialization(error.to_string()))?;
        let digest = Sha256::digest(bytes);
        Ok(SpecDigest(format!(
            "sha256:{}",
            digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        )))
    }

    pub fn diff(&self, other: &Self) -> Result<AgentDiff> {
        self.validate()?;
        other.validate()?;
        AgentDiff::between(self, other)
    }
}
