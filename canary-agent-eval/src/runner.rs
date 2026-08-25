use crate::environment::{EnvironmentStatus, EvalEnvironment};
use crate::error::{EvalError, Result};
use crate::roles::{EvalReport, EvaluatedPolicy, Referee, RefereeInput};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvalRunnerConfig {
    pub max_environment_steps: usize,
}

impl Default for EvalRunnerConfig {
    fn default() -> Self {
        Self {
            max_environment_steps: 128,
        }
    }
}

pub struct EvalRunnerComponents {
    pub policy: Arc<dyn EvaluatedPolicy>,
    pub referee: Arc<dyn Referee>,
}

impl EvalRunnerComponents {
    pub fn new<P, R>(policy: P, referee: R) -> Self
    where
        P: EvaluatedPolicy + 'static,
        R: Referee + 'static,
    {
        Self {
            policy: Arc::new(policy),
            referee: Arc::new(referee),
        }
    }
}

pub struct EvalRunner {
    config: EvalRunnerConfig,
    environment: Box<dyn EvalEnvironment>,
    components: EvalRunnerComponents,
}

impl EvalRunner {
    pub fn new<E>(environment: E, components: EvalRunnerComponents) -> Self
    where
        E: EvalEnvironment + 'static,
    {
        Self {
            config: EvalRunnerConfig::default(),
            environment: Box::new(environment),
            components,
        }
    }

    pub fn with_config(mut self, config: EvalRunnerConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn run(&mut self) -> Result<EvalReport> {
        let mut output = self.environment.reset().await?;
        let mut steps = 0usize;
        while output.status == EnvironmentStatus::Running {
            if steps >= self.config.max_environment_steps {
                return Err(EvalError::Environment(format!(
                    "environment exceeded {} steps",
                    self.config.max_environment_steps
                )));
            }
            let observation = output.observation.take().ok_or_else(|| {
                EvalError::Environment(
                    "running environment did not produce an observation".to_string(),
                )
            })?;
            let action = self.components.policy.act(observation).await?;
            output = self.environment.step(action).await?;
            steps = steps.saturating_add(1);
        }

        self.components
            .referee
            .evaluate(RefereeInput {
                snapshot: self.environment.snapshot()?,
                trajectory: self.environment.trajectory(),
            })
            .await
    }

    pub fn environment(&self) -> &dyn EvalEnvironment {
        self.environment.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::{
        EnvironmentController, EnvironmentControllerInput, EnvironmentDecision, EnvironmentFuture,
        GraphEnvironment, ObservationContent, ObservationRealizer, ObservationRealizerInput,
    };
    use crate::program::{
        NodeId, TaskCase, TaskNode, TaskTransition, TransitionId, TransitionKind,
    };
    use crate::roles::{
        ActionFuture, AgentAction, AgentActionStatus, EvalReportFuture, RefereeInput,
    };
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Policy {
        calls: Arc<AtomicUsize>,
    }

    impl EvaluatedPolicy for Policy {
        fn act<'a>(
            &'a self,
            observation: crate::EnvironmentObservation,
        ) -> ActionFuture<'a, AgentAction> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move {
                Ok(AgentAction {
                    status: AgentActionStatus::Completed,
                    assistant_text: observation.user_text,
                    events: Vec::new(),
                })
            })
        }
    }

    struct Controller;

    impl EnvironmentController for Controller {
        fn decide(
            &self,
            _input: EnvironmentControllerInput,
        ) -> EnvironmentFuture<'_, EnvironmentDecision> {
            Box::pin(async {
                Ok(EnvironmentDecision::Transition {
                    transition: TransitionId::from("finish"),
                    evidence: Vec::new(),
                    reason: "done".to_string(),
                })
            })
        }
    }

    struct Realizer;

    impl ObservationRealizer for Realizer {
        fn realize(
            &self,
            _input: ObservationRealizerInput,
        ) -> EnvironmentFuture<'_, ObservationContent> {
            Box::pin(async {
                Ok(ObservationContent {
                    user_text: "perform task".to_string(),
                    exposures: Vec::new(),
                    metadata: Value::Null,
                })
            })
        }
    }

    struct TestReferee;

    impl Referee for TestReferee {
        fn evaluate(&self, input: RefereeInput) -> EvalReportFuture<'_, EvalReport> {
            Box::pin(async move {
                assert_eq!(input.snapshot.status, EnvironmentStatus::Terminated);
                assert!(!input.trajectory.is_empty());
                Ok(EvalReport {
                    metrics: Vec::new(),
                    overall_score: Some(1.0),
                    details: json!({}),
                })
            })
        }
    }

    fn environment() -> GraphEnvironment {
        let graph = TaskCase {
            id: "runner".to_string(),
            version: "1".to_string(),
            start: NodeId::from("start"),
            nodes: vec![
                TaskNode {
                    id: NodeId::from("start"),
                    constraints: Vec::new(),
                    result_obligation: None,
                    evidence_obligation: None,
                    terminal: false,
                },
                TaskNode {
                    id: NodeId::from("done"),
                    constraints: Vec::new(),
                    result_obligation: None,
                    evidence_obligation: None,
                    terminal: true,
                },
            ],
            transitions: vec![TaskTransition {
                id: TransitionId::from("finish"),
                from: NodeId::from("start"),
                to: NodeId::from("done"),
                kind: TransitionKind::Progress,
            }],
        }
        .compile()
        .expect("graph");
        GraphEnvironment::new(graph, Controller, Realizer).expect("environment")
    }

    #[tokio::test]
    async fn runner_only_coordinates_environment_policy_and_referee() {
        let calls = Arc::new(AtomicUsize::new(0));
        let components = EvalRunnerComponents::new(
            Policy {
                calls: calls.clone(),
            },
            TestReferee,
        );
        let mut runner = EvalRunner::new(environment(), components);
        let report = runner.run().await.expect("run");

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(report.overall_score, Some(1.0));
    }
}
