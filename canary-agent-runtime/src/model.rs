use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;

pub use canary_agent_kernel::model::{
    FunctionSpec, ModelFunctionCall, ModelRequest, ModelResponse, ModelStreamEvent,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub fqn: String,
    pub settings: serde_json::Value,
}

pub type ModelStreamHandler<'a> = dyn FnMut(ModelStreamEvent) + Send + 'a;

pub trait ModelClient: Send + Sync {
    fn model_descriptor(&self) -> ModelDescriptor;

    fn stream_complete<'a>(
        &'a self,
        request: ModelRequest,
        on_event: &'a mut ModelStreamHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = crate::Result<ModelResponse>> + Send + 'a>>;
}
