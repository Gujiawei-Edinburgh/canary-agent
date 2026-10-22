# Canary ReAct Agent Runtime

## Core Idea

An active interaction does not have exclusive ownership of its input history.
Users and other sources can submit new events while the agent is reasoning,
streaming a response, or running tools.

Canary keeps the ReAct model: the model interprets new information and chooses
responses or tool calls. The runtime records events, coordinates inference,
tracks tool operations, and streams output. Tools own their operational behavior.

Audio interaction, robotics, coding, and remote jobs are applications of the
same execution model. The SDK exposes its single execution engine through
`Agent::open_turn`. Successful completion requires `end_turn()`;
the host observes the committed completion event and decides what happens next.

## From Sequential To Full Duplex

The former sequential loop ran consecutive iterations within a turn. Each iteration
infers a response or actions, executes the selected tools, and records results
before the next iteration starts:

```text
iteration A: infer -> tools -> results
iteration B:                          infer -> tools -> results
```

Full duplex allows input and output simultaneously. Long-running tools no longer
prevent a new inference from responding to new input:

```text
input:       request A          refinement B              input C
inference:   infer A            infer B                   infer C
tools:            operation A --------------------------> outcome A
                                     operation B -> outcome B
output:      response deltas and operation events as they become available
```

The initial policy allows one authoritative primary inference at a time and
multiple active tool operations. The overlap is between inference, tool
execution, and event ingestion; it does not require multiple authoritative
primary-model responses racing to control the interaction.

## History And Ownership

A thread remains durable interaction history. An open turn becomes an
interaction episode that may contain several user inputs, assistant responses,
and overlapping tool operations. A session remains the live runtime attached to
that history; broadening a turn does not require merging these concepts.

Applications submit events through ingress rather than mutate the item list.
The runtime orders and persists those events through one logical writer per
interaction. Inference and tools run outside that writer, so waiting for them
does not block incoming events.

The history is factual and append-only:

```text
UserInput
ModelResponse(tool call A)
ToolAccepted(call A, operation A)
UserInput
ModelResponse(tool call B)
ToolAccepted(call B, operation B)
ToolCompleted(operation A, outcome)
```

Each inference records the history position it consumed. Tool calls and
operations have correlated identities. Active-operation state is derived from
their lifecycle events; live cancellation handles are execution state.

## New Input During Inference

New user input supersedes the current inference:

1. Record the input and mark the current inference superseded.
2. Request cancellation of its model call.
3. Build context from the updated history and start a replacement inference.

Cancellation is best effort. A superseded inference cannot publish further
response deltas or dispatch tool calls, even if it finishes late. Event
processing orders supersession against acceptance of a completed model response.
Calls already accepted for dispatch become tracked operations and follow the
tool-execution rules below.

Text or audio already delivered cannot be retracted. The runtime emits an
interruption event so the presentation layer can stop playback or indicate that
the response was interrupted.

Ingress receives meaningful input events. Audio segmentation belongs upstream;
raw frames should not continuously restart inference. Scheduling may coalesce
pending triggers while preserving their recorded inputs.

## New Input During Tool Execution

New input does not automatically cancel active tools. The runtime records it and
starts inference using the updated history and an explicit view of active
operations. The model can continue existing work, add work, or call a tool that
requests cancellation or modification of a specific operation.

For example, changing a delivery item need not cancel travel to its location.
Understanding that relationship belongs to the model, not a task classifier
inside the runtime.

A tool reports acceptance followed by a terminal outcome. Progress is optional:

```text
accepted -> [zero or more progress events] -> terminal outcome
```

Admission may instead reject a call. An immediate tool may return its terminal
result directly without a separate asynchronous acceptance receipt.

Acceptance means the operation has been admitted, not that it succeeded.
Completion appends a new event; it never replaces the acceptance or progress
events. A failure is also a factual outcome. Retrying it creates a new call and
operation rather than rewriting the previous attempt.

Requesting cancellation and completing cancellation are distinct. The original
operation reports what actually happened: it may stop, finish before the request
takes effect, or fail to stop. Late tool outcomes remain facts even when the
request that motivated them has been superseded.

Tools own resource-specific serialization and stopping behavior. The runtime
owns operation tracking and generic capacity limits. Control calls must be able
to proceed while ordinary execution capacity is occupied.

## Model Context And Protocol

The journal and the model's message list are different representations.
Context construction includes relevant history and active-operation state,
even when older conversation content is compacted.

For the proposed Chat Completions adapter, an asynchronous tool's acceptance
receipt answers the protocol-level tool call. Its operation continues
independently. Every call in an assistant tool-call batch receives an acceptance,
immediate result, or rejection before that batch is used in another model
request. This protocol requirement does not block recording incoming events.

Later progress and completion are runtime observations, not additional protocol
responses to an already answered tool-call ID. The adapter appends them as
clearly labeled user-role messages because this protocol has no observation
role. Internally, they remain tool events with their original provenance.

An illustrative completion envelope is:

```json
{
  "source": "runtime",
  "event": "tool_completed",
  "operation_id": "op_a",
  "originating_call_id": "call_a",
  "outcome": {
    "status": "succeeded",
    "output": {}
  }
}
```

The runtime constructs the envelope. Model instructions distinguish these
observations from human requests and treat tool content as data, not
instructions. The tool interface explains that acceptance is not completion.
The adapter preserves event provenance and does not rewrite earlier receipts
or invent successful results to satisfy the protocol.

This envelope is an application-level encoding, not a native asynchronous
tool-message feature. Other model adapters may use a native event representation.
Provider-specific mappings require compatibility tests.

## Waiting And Closing

A text-only response ends the current reasoning run. It does not automatically
close an open turn or trigger another inference. The runtime waits for new input
or an operation outcome; tools can remain active during that wait.

User input supersedes active inference. Tool completions are recorded and
schedule further inference; if inference is already running, the initial policy
queues a continuation using the updated history. Progress is recorded when
provided, but does not trigger inference by default. Several pending triggers
may be served by one inference that observes them all.

The first-party `end_turn()` command explicitly closes the turn.
Although model-visible as a tool, it is runtime-owned because it changes the
interaction lifecycle.

The runtime accepts closure only when no operations or unhandled inputs and
outcomes remain. It checks this against current state when processing the
command, not merely the snapshot used by the model. New input or an unseen tool
outcome therefore prevents a stale inference from closing the turn. A batch
containing both closure and new operation requests cannot close the turn.

If closure is rejected, the command returns the reason to the model. Closure
does not implicitly cancel tools or discard events. Once closure is accepted,
later user input belongs to a new turn according to the host's interaction policy.
Host abort and shutdown remain separate lifecycle operations.

## Boundaries And Compatibility

The runtime owns event ordering, persistence, inference supersession, streaming,
tool-call correlation, operation tracking, context adaptation, and turn closure.
The model decides how instructions relate and what actions to request. Tools
own their domain behavior, authorization enforcement, resource rules, and
downstream cancellation.

The runtime does not require semantic task graphs or a separate intent model.
Audio capture, endpoint detection, playback, and hard real-time safety remain
outside the loop. A slow reasoning model can be exposed as a tool; this does not
require the runtime to classify task complexity.

Full duplex is the only runtime engine. A host can submit a single input and
wait for completion or keep submitting input while work is running. Both use
the same explicit closure, cancellation, operation, and recovery semantics.
A text-only response never closes a turn; the model can include final answer
text and an end_turn call in the same response.

The first implementation should demonstrate four cases: superseded inference
cannot dispatch late calls; new input can control a running operation; completion
appends without changing acceptance; and concurrent input prevents stale closure.
Detailed persistence, recovery, and provider contracts follow from those cases.

The initial implementation also exposes `recover_turn`. Recovery
reconstructs operations from acceptance and terminal events, retries only
idempotent unfinished executions automatically, and leaves other outcomes
indeterminate for explicit host reconciliation. Tool suspensions are tracked
per operation and resolved through the live interaction handle. Cancellation
and execution timeouts preserve unknown external outcomes when the executor
cannot confirm completion.
