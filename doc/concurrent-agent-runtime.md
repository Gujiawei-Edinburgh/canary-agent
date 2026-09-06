# Concurrent Agent Runtime

## Purpose

Canary currently runs a typical agent loop: receive an input, run one model
and tool iteration, commit its result, and then start the next iteration. This is
a useful sequential agent, but it is not a general execution model for an
environment that changes while the agent is working.

This document describes the architectural shift from sequential loop iterations
to overlapping work pipelines. Detailed APIs and state machines belong in later
documents.

## Why The Loop Must Change

General-purpose agents coordinate components with very different frequencies. A
remote reasoning model may take seconds. A local policy may run several times
per second. Sensors and controllers may update hundreds or thousands of times
per second.

The environment also has independent writers: users, other agents, services,
devices, and the physical world. State can change while a model is reasoning.

A request-response loop serializes this work:

```text
input A -> reason A -> act A -> input B -> reason B -> act B
```

This shape works when one interaction owns the environment and changes arrive
between turns / iterations. It blocks fast or urgent work behind slow work when those
assumptions do not hold.

## The Core Shift

One iteration of the current agent loop can be viewed as a five-stage pipeline:

```text
observe -> compute -> propose -> validate -> retire
```

Today, a new iteration starts after the previous iteration finishes:

```text
cycle      0   1   2   3   4   5   6   7   8   9

iter A     O ->C ->P ->V ->R
iter B                         O ->C ->P ->V ->R
```

Canary generalizes the loop by allowing independent pipelines to overlap:

```text
cycle      0   1   2   3   4   5   6

work A     O ->C ->C ->C ->P ->V ->R
work B         O ->C ->P ->V ->R
work C             O ->P ->V ->R
```

A cycle is a logical scheduling opportunity, not a fixed unit of wall-clock
time. A stage may take many cycles or complete immediately.

> Canary admits new work while earlier work is still running. It tracks the
> assumptions behind every in-flight pipeline, commits valid results
> independently, and flushes superseded work without stopping unrelated work.

## The Common Pipeline

Inputs from users, sensors, timers, tools, services, and other agents are first
recorded as factual events. Admission policy decides whether an event creates
work and where that work is dispatched. An event that does not create work does
not start a pipeline. Every admitted work item follows the same path:

- **Observe** captures the environment view and initial dependencies used by the
  work.
- **Compute** produces a result through inference, perception, planning, a tool,
  or another worker.
- **Propose** places the candidate result in the in-flight work window.
- **Validate** checks that its dependencies and applicable guards still hold.
- **Retire** ends the pipeline with a `Committed` or `Discarded` outcome.

Stages that do not apply are zero-cost transitions. A deterministic rule may
pass through compute immediately. A frontier-model call may remain in compute
for many cycles.

An input event and a work item are distinct. An event may update runtime
knowledge, invalidate existing work, and admit zero or more new work items.

Compute failure does not send the same work backward through the pipeline. A
tool or function failure is a factual result that may retire as `Committed`. If
policy chooses to retry it, that decision admits a new work item and therefore a
new pipeline.

## The In-Flight Work Window

Overlapping pipelines require a central registry of work that has been admitted
but not retired. This is Canary's analogue of a reorder buffer (ROB). It records:

- work identity and status;
- the observations or projections the work consumed;
- dependencies on earlier work;
- authority over affected resources;
- completed proposals waiting for validation.

Unlike a CPU ROB, Canary does not require one global retirement
order. Independent work may retire as soon as it is valid. Explicit dependencies
and resource ordering determine when ordering is required.

## Dependencies And Pipeline Flushes

When work consumes runtime knowledge, it records a read set such as:

```text
human/intent       @ version 7
object/cup/pose    @ version 42
navigation/map     @ version 12
```

The admission policy and context builder declare known dependencies. Workers
may report additional reads discovered while running. A witness can be a
version, hash, sequence, timestamp, or application-defined condition.

When new input changes one of these assumptions, Canary invalidates the affected
work and its dependants. It requests cancellation of running computation,
blocks late proposals from committing, and may admit replacement work. Pipelines
with unrelated dependencies continue.

Cancellation saves resources. Validation at retirement provides correctness
because cancelled work may still finish and return a result.

## Four Typical Cases

**Independent work:** two pipelines read unrelated state. Both compute and
commit concurrently.

**Stale work:** a plan reads a cup pose at version 42. Version 43 arrives before
retirement. Canary flushes that plan and may issue a replacement.

**Failed compute:** a tool fails and that factual failure commits. Retry policy
admits another work item instead of moving the failed pipeline backward.

**Dependent work:** work B depends on a proposal from work A. If A is discarded,
B is flushed while pipelines with unrelated dependencies continue.

## The Current Agent As A Special Case

The existing turn runtime is this architecture with one active pipeline:

- one primary input channel;
- an in-flight capacity of one;
- model calls and function calls as workers;
- thread state as the primary projection;
- sequential validation and retirement;
- user cancellation as invalidation.

Within a turn, each model iteration waits for the preceding model and function
results before it starts. Reproducing this behavior with pipeline capacity one
is the first compatibility test for the general runtime. Increasing capacity and
adding dependency-aware retirement then enables concurrent operation without
requiring a separate agent architecture.

## Boundary

Canary owns admission, in-flight work, dependency tracking, scheduling,
validation, retirement, cancellation, and recovery.
Applications own projections, goals, decision policies, dependency meaning,
resource authority, safety rules, and workers.

Fast safety and control loops remain outside slow model inference. Canary may
send them validated goals or commands, but it does not provide perception
algorithms, robotics middleware, motor control, hard real-time guarantees, or
model training.

## Next Step

Define the minimal work lifecycle, in-flight registry, dependency contract, and
retirement invariants. Implement the existing agent loop as the capacity-one
profile before enabling overlapping pipelines.
