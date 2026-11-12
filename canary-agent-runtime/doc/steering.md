# Cooperative Steering

The request runtime follows the sequential ReAct shape: build context, sample
once, execute the returned tool calls, and repeat. Steering adds an input queue;
it does not add speculative model requests or preempt current work.

```text
iteration 1: drain input -> context -> model -> tool batch
                              new input arrives -> queue
iteration 2: drain queued input -> context -> model -> ...
```

## Boundary Semantics

- An iteration boundary is before context construction, after the preceding
  inference and complete tool batch, including its hooks, have returned.
- Queued inputs are recorded together, in FIFO order, as factual `UserInput`
  items in the same turn. Only then is the next model context built.
- An inference already in flight finishes against its original snapshot. Its
  response is retained, not discarded because newer input exists.
- Requested tool calls still execute. A queued refinement does not implicitly
  cancel or rewrite them. The next inference decides what to do with new facts.
- A response without tool calls ends the turn only if the queue is empty.
  Otherwise its response is recorded and another iteration processes steering.
- Iteration and function-call limits apply to the entire turn, including all
  steering-driven follow-ups. Steering does not reset the budgets.

These are scheduling rules, not business decisions. The model interprets the
meaning of refinements. Urgent cancellation remains the explicit `abort` path,
which can interrupt inference or tools independently of the steering queue.

## SDK Interface

Run `agent.run_turn(...)` in one task. After observing its `TurnStarted` event,
use the same agent or a clone to steer the active thread:

```rust
let receipt = agent.steer("thread-id", "Focus on cancellation, not formatting")?;
let input_id = receipt.input_id.clone();
receipt.wait_recorded().await?;
```

`steer` returns immediately after local queue admission. `wait_recorded` waits
for the next boundary's durable commit. Neither confirms that the model has
consumed the input or completed the requested work. Dropping a receipt does not
withdraw the queued input.

The queue is per active turn and bounded to 64 inputs by default. Configure it
before starting turns with `with_steering_capacity(NonZeroUsize)`. A full queue
returns `SteeringQueueFull` rather than silently dropping input.

## Completion And Failure

Queue admission and completion sealing share a mutex. If steering wins the race,
the loop cannot complete successfully without processing another iteration. If
completion wins, steering returns `SteeringClosed`; after the active turn is
released it returns `TurnNotActive`. Input never silently migrates to a later turn.

Abort, failure, suspension, or dropping the running future closes the queue and
rejects unconfirmed receipts. Pending input is memory-only and is not recovered
after a process crash. Once committed, it is part of normal durable history.
On an uncertain storage error, use the receipt's `input_id` to inspect the
journal; a missing acknowledgment is not proof that an external store wrote
nothing.

Recovery and suspension resolution keep their existing execution semantics.
Steering queued during recovery or resumed execution waits for the next normal
iteration boundary. It cannot authorize a suspended function; authorization or
reconciliation remains an explicit host action.

## Scope

This implementation builds on the sequential runtime copied from `feat-steer`,
on `main`. It does not restore the former `RunningTurn`, bidirectional-session,
or overlapping-operation APIs. Crates and examples still using those APIs need
a separate integration migration; this queue is not a realtime transport adapter.
