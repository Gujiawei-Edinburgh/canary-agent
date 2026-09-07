# Concurrent Agent Runtime

## Purpose

Canary currently runs a sequential ReAct-style agent loop: receive an input,
infer a response or actions, execute tools, record the results, and infer again
when needed. This is useful for conversational and coding agents, but the
environment may change while either inference or tool execution is running.

This document describes the shift to **pipelined ReAct**: multiple inference and
execution pipelines can overlap, responding to new events while earlier work
remains active. It establishes the high-level model and its surrounding event
flow. Detailed contracts, APIs, and state machines belong in a later document.

## Why The Loop Must Change

General-purpose agents coordinate components with very different frequencies. A
remote reasoning model may take seconds. A local policy may run several times
per second. Sensors and controllers may update hundreds or thousands of times
per second. A tool may also remain active for a long time, such as an arm moving
toward an object or a remote service carrying out a job.

The environment has independent writers: users, other agents, services, devices,
and the physical world. New information can require a response before earlier
work finishes.

A sequential loop processes work like this:

```text
input A -> infer A -> execute A -> result A -> process next input
```

If new work must wait for that sequence, fast or urgent responses can be blocked
behind slow inference or long-running actions. Pipelined ReAct allows new work
to start while other pipelines are inferring or executing.

## The Overall Flow

```text
Sources -> event ingress -> projection -> admission
                                ^             |
                                |             v
                          outcome events <- [infer -> execute]
                                |
                                v
                        external subscribers
```

Sources include users, sensors, timers, services, tools, and other agents. Event
ingress records incoming facts. The environment projection incorporates those
facts into the view available to subsequent decisions. Trigger and admission
policy determines whether an event should create work and what work to create;
the runtime schedules admitted work within its limits.

An event and a pipeline are distinct. An event may update the projection without
starting any pipeline, or it may lead to multiple pipelines. Admission considers
the updated environment view rather than the view preceding the triggering event.

Outcome events feed back into the same projection and admission flow. External
subscribers can also observe them for presentation, monitoring, or integration.
A completed pipeline does not unconditionally start another one: policy decides
whether its outcome calls for further work.

Changing the environment and updating its projection are distinct. A tool can
change the physical or external environment while it is executing. The
projection changes when events report observations or execution outcomes; it is
the runtime's available view, not the environment itself. Independent sources
can update that view before a running tool finishes.

## The Common Pipeline

Each admitted pipeline receives its input and a captured environment view, then
follows this lifecycle:

```text
infer -> execute
```

- **Infer** uses the input and environment view to produce a response or actions.
  The inference engine may be rule-based, VLA, WAM, LLM, or another implementation.
  A direct action request still passes through inference: a deterministic rule
  maps the request to an action without requiring a model call.
- **Execute** runs the selected tools or actions. This stage remains active for
  the duration of execution, including long-running operations.

A response without actions requires no tool execution. Either stage can end
with a failure or abort. These are forward transitions; retrying or reasoning
about a failure admits a new pipeline rather than sending the same pipeline
backward.

Completion is a lifecycle transition, not a third processing stage. The pipeline
produces an outcome; the surrounding runtime records completion and publishes
the outcome event. Projection updates, further admission, and subscriber
delivery happen outside the pipeline stages, within the surrounding event flow.
These responsibilities remain part of the SDK even though they are not stages.

Observation is the input to this lifecycle, not a mandatory standalone stage.
The runtime must retain the association between a pipeline and the environment
view it consumed so that its decisions can be understood later. Capturing and
constructing that view belongs to the input and projection contracts.

## Overlapping Pipelines

Independent pipelines can be active at different stages:

```text
time -------------------------------------------------------->

pipeline A    infer -> execute --------------------------> outcome
pipeline B             infer -> execute -> outcome
pipeline C                         infer -> execute -> outcome
```

Pipeline B can react to information that arrived after A started. Pipeline C
can use deterministic inference to select an action immediately. The outcomes
mark completion, not additional pipeline stages. Independent pipelines
do not need to finish in admission order; scheduling and resource rules govern
which executions can overlap.

The runtime tracks active pipeline identities, their input views, execution
handles, status, and outcomes. It uses this information to route results and
cancellation requests to the correct work. It does not need to understand a
tool's internal execution process.

## Tool Execution And Cancellation

A tool executor owns the details of carrying out an action. For an arm movement,
the pipeline sees execution start and eventually an outcome; movement planning,
device communication, and control remain within the tool's implementation. The
pipeline stays in the execution stage while the arm is moving.

Consider a cup being removed while the arm approaches it:

1. Pipeline A infers an action and starts the arm movement.
2. A new event reports that the cup was removed, updating the environment view.
3. Policy admits pipeline B. Its inference engine selects cancellation, and its
   execution requests cancellation of A's execution.
4. The executor handles that request and reports the actual outcome to A.
5. A completes with that outcome. The runtime records completion and publishes
   the outcome event. Policy may use it and the updated environment view to
   admit another pipeline.

Requesting cancellation and completing cancellation are different events. A
request can be accepted while the original execution is still stopping. A tool
may finish before cancellation takes effect or report that stopping failed. The
original pipeline must reflect the executor's actual outcome, not assume that
sending a cancellation request immediately aborted execution.

Cancellation does not undo effects that have already occurred. Outcomes must
preserve known execution facts, including failures and partial effects, even when
the original action is no longer useful. A late inference result must not start
new actions after its pipeline has been cancelled, while a late execution result
may still provide facts about what happened.

Scheduling must allow cancellation handling to proceed while ordinary execution
capacity is occupied. Otherwise, the work intended to stop a running action could
be blocked behind that action.

## Responding To A Changing Environment

There is no mandatory validation stage between inference and execution. A check
at one moment cannot establish that an action remains appropriate throughout its
execution. New events can lead to new decisions, cancellation, and replacement
work while earlier pipelines remain active.

Authorization, resource ownership, and execution guards still belong in the
relevant contracts. They constrain whether and how an action starts or continues;
they do not claim that the environment will remain unchanged. Policies may use
the view consumed by a pipeline and its relationship to other work to decide
what should be cancelled or replaced.

Fast safety and control loops remain outside slow inference. Event-driven
cancellation complements those mechanisms; it does not provide an instantaneous
stop or a hard real-time safety guarantee.

## The Current Agent As A Special Case

The existing conversation runtime maps to a simple sequential profile of
pipelined ReAct. Its simplifications cover the surrounding event flow as well
as the absence of overlapping pipelines:

| Part | Current sequential profile |
| --- | --- |
| Sources and ingress | User input and model/tool results are recorded as thread facts. |
| Environment-to-projection propagation | Reported facts update the thread, from which the conversation view is derived. There is no separate continuous environment synchronization loop. |
| Trigger and admission | Fixed loop control starts inference for user input, continues after tool results when appropriate, and ends the turn for a response without tool calls. |
| Inference and execution | The model selects responses or actions; registered functions execute actions. |
| Concurrency | Within a turn, the next inference waits for the preceding inference and tool executions; pipelines do not overlap. |

Projection propagation and admission are therefore effectively dummy adapters
in this profile: direct propagation of reported facts and fixed continuation
rules. "Dummy" means minimal policy, not an absence of projection logic or
recording. The current implementation embeds these behaviors in the agent loop;
they are not yet separate general-purpose runtime interfaces.

For example, a tool changes an external environment and returns a result. The
loop records the result, derives the updated conversation view, and continues
inference. It learns about the change through the reported result, without an
independent sensing and reconciliation path. A general-purpose integration can
instead receive observations while that same tool is still executing and admit
another pipeline in response.

A turn can therefore span multiple pipelines. Each inference and its selected
tool executions form one pipeline, whose outcome may trigger the next. Capacity
one alone does not define conversational behavior; the trigger and admission
policy also determines which work follows each outcome.

Reproducing the current agent's behavior with this profile is the first
compatibility target. Additional sources and concurrent admission then extend
the same execution model to other kinds of agents.

## Runtime And Integrator Boundary

The **runtime** means the SDK provided by this project, including its execution
engine, public contracts, and built-in implementations. The **application** means
the integrator's code that configures the SDK and supplies implementations for
its environment. An injected implementation can run inside the same process and
participate directly in runtime execution.

The SDK defines the contracts and coordinates event processing, admission,
scheduling, pipeline lifecycle, execution tracking, cancellation, outcome
publication, and recovery. It may provide reusable implementations of those
contracts.

Integrators connect event sources and supply domain-specific projection logic,
goals, trigger policies, inference providers, tools, resource rules, and safety
mechanisms as needed. The runtime controls the lifecycle through these contracts;
it does not need to know how a particular tool performs its action or what a
particular environment fact means.

Canary does not provide perception algorithms, robotics middleware, motor
control, hard real-time guarantees, or model training. Such components can be
integrated as sources, inference providers, or tool executors.

## Pseudocode
```rust
while let Some(event) = event_ingress.next().await {
    projection.apply(&event);

    for pipeline in admission.issue(&event, &projection) {
        tokio::spawn(async move {
            let actions = pipeline.infer().await;
            let outcome = executor.execute(actions).await;
            publish(Event::Outcome(outcome)).await;
        });
    }
}
```
