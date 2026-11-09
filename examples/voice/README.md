# Voice Assistant

## Architecture

See [Voice Agent Architecture](doc/architecture.md) for the event-driven loop,
its comparison with serialized ReAct, and provider support boundaries.

Run `cargo run -p voice -- --p qwen`. The single voice agent includes reasoning
delegation alongside its domain tools. `delegate` makes one reasoning-model request and returns
advice and proposed calls; it does not run a separate worker agent or execute
those proposals. Domain tools remain directly available to the foreground model.
Set `CANARY_AGENT_MODEL` and `CANARY_AGENT_API_KEY` for the reasoning model.
Optionally set `CANARY_AGENT_BASE_URL` and `CANARY_AGENT_REASONING_EFFORT`.
There is no separate `live` binary or `--live` mode.

A continuous audio example built on Canary. It uses the official DashScope Python
SDK through a small IPC bridge; Qwen-specific code lives only in this example.

```text
microphone -> Rust audio/AEC -> Canary -> Rust BridgeClient
                                           | JSONL stdin/stdout
                                       Python worker -> DashScope SDK -> Qwen
speaker <- Rust playback <------------------|
```

Rust owns microphone capture, acoustic echo cancellation, playback, tools, and the
Canary session. Python owns SDK connection/configuration and translates SDK
callbacks into example IPC events. Audio crosses IPC as base64 PCM16. Python
logs go to stderr, not the JSON event stream.

## Run

From the workspace root:

```bash
python3 -m venv examples/voice/.venv
examples/voice/.venv/bin/python -m pip install -r examples/voice/python/requirements.txt
export VOICE_PYTHON="$PWD/examples/voice/.venv/bin/python"

cargo run -p voice --bin voice -- --check-config
cargo run -p voice --bin voice -- --check-connection
cargo run -p voice --bin voice
```

Fill in `examples/voice/.env` using `.env.example`. Existing environment variables
take precedence. `--check-config` only validates local settings.
`--check-connection` starts the Python SDK and configures the remote session
without opening audio devices or requesting inference.

Required Qwen settings: `REAL_TIME_ENDPOINT` and `REAL_TIME_API_KEY`.
The unified tool registry also requires `CANARY_AGENT_MODEL` and
`CANARY_AGENT_API_KEY`; missing reasoning configuration fails before audio opens.
`--check-config` validates these settings and lists configured tools, including
`delegate`. `--check-connection` submits the same prompt and tool schemas as the
normal application, but does not test reasoning-model inference or tool execution.
Defaults: `REAL_TIME_MODEL=qwen3.8-omni-flash-realtime`,
`REAL_TIME_VOICE=Tina`. Python defaults to the example's `.venv/bin/python` when
present, otherwise `python3`; `VOICE_PYTHON` overrides this choice.
Use your workspace-specific Qwen endpoint. HTTP bases ending in
`/compatible-mode/v1` are normalized to `wss://.../api-ws/v1/realtime`.
The SDK appends the model query itself. Select Qwen with `--p qwen` (the default).

For MOSS audio conversation, set `MOSS_API_KEY`, install
`python/requirements-moss.txt` into the same Python environment, and use
`cargo run -p voice --bin voice -- --p moss`. Its adapter does not currently
support Canary custom tools or typed input. Configuring `delegate` locally does
not make it usable through that adapter; startup warns about this limitation.
See the architecture document for the capability boundary.

## Interaction

Grant microphone access to your terminal on macOS. Speak after `[listening]`.
The microphone stays active during speech and tools. Server VAD detects speech
boundaries. New speech does not automatically clear buffered playback. Tools
continue unless cancelled.

Playback queues responses in arrival order without replacing older audio. The
model can call `get_playback_state` for playback IDs, generated-text previews,
and estimated cursors, then `stop_playback({"playback_id":"..."})` to discard one
specific response. Late chunks for that stopped response are ignored; other
responses remain queued. State queries are ordinary journaled tool calls, not
continuous feedback or native Qwen audio-history truncation.

Explicit provider playback-stop events still apply. Qwen can interrupt server
generation independently; preserving locally received audio does not prevent
that service-side behavior. Model-directed stopping has inference latency. Audio
already submitted to the device cannot be recalled; cursor values are estimates,
not proof of which words a person heard.

AEC3 uses actual speaker output as an echo reference, processing 10 ms frames at
48 kHz. Input is resampled to Qwen's mono 16 kHz PCM16; output is 24 kHz PCM16.
High volume can still leave residual echo; reduce volume or use headphones.

Type a message and press Enter to send text into the same session as the
microphone. Text is journaled before delivery and does not automatically cancel
speech or tools. It is delivered immediately; response requests wait for the
current response/tool cycle and speech detection to settle. Provider text support
depends on the selected model. Text is not echoed as a second input transcript.

Local commands:

- `/mute`, `/unmute`: microphone forwarding.
- `/stop`: immediately discard current and queued local playback, without
  cancelling model generation or other operations.
- `/reconnect`: open a new SDK session after a failure.
- `/quit` or Ctrl+C: abort the interaction and its tools.

The model calls `end_turn()` when the conversation ends. Final playback drains
before exit. Reconnection restores selected textual history as labeled context,
but waits for fresh speech or text instead of issuing inference against an empty audio
session. Audio is never replayed and completed tools are never re-executed.
Late results from an old session update context rather than using old call IDs.

## Tools And State

The demo registers delegation, playback observation/control, current time, and
sandboxed `exec_command`. Set `EXA_API_KEY`
for `web_search`. `VOICE_WORKSPACE` defaults to the current directory; sandbox
escalation is denied. `VOICE_THREAD_ID` reuses a thread.
Logs, traces, and thread JSON live in `.canary-agent-voice/` unless
`VOICE_STATE_DIR` is set. Raw audio is not journaled. Token totals print on exit.

Provider errors are surfaced by the worker. For `AllocationQuota.FreeTierOnly`,
review the provider's quota and payment settings. Disabling Free Quota Only can
enable paid usage.

## Tests

These tests use mock SDK callbacks and a local Python process, not live APIs or
audio devices. They do not establish live-provider compatibility.

```bash
python3 -m unittest discover -s examples/voice/python -v
cargo test -p voice -p canary-agent-openai
```

Linux audio builds need `libasound2-dev` and `pkg-config`; macOS uses CoreAudio.
OpenAI protocol support remains in `canary-agent-openai`, independent of this
example.

Reference: [official DashScope Python SDK](https://help.aliyun.com/en/model-studio/omni-realtime-python-sdk).
