# Qwen Voice Assistant

A continuous audio example built on Canary. It uses the official DashScope Python
SDK through a small IPC bridge; Qwen-specific code lives only in this example.

```text
microphone -> Rust audio/AEC -> Canary -> Rust QwenClient
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

cargo run -p voice -- --check-config
cargo run -p voice -- --check-connection
cargo run -p voice
```

Fill in `examples/voice/.env` using `.env.example`. Existing environment variables
take precedence. `--check-config` only validates local settings.
`--check-connection` starts the Python SDK and configures the remote session
without opening audio devices or requesting inference.

Required settings: `REAL_TIME_ENDPOINT` and `REAL_TIME_API_KEY`.
Defaults: `REAL_TIME_MODEL=qwen3.8-omni-flash-realtime`,
`REAL_TIME_VOICE=Tina`. Python defaults to the example's `.venv/bin/python` when
present, otherwise `python3`; `VOICE_PYTHON` overrides this choice.
Use your workspace-specific Qwen endpoint. HTTP bases ending in
`/compatible-mode/v1` are normalized to `wss://.../api-ws/v1/realtime`.
The SDK appends the model query itself. This demo is Qwen-only;
`REAL_TIME_PROTOCOL` can be `qwen` or omitted.

## Interaction

Grant microphone access to your terminal on macOS. Speak after `[listening]`.
The microphone stays active during speech and tools. Server VAD detects speech
boundaries; new speech clears buffered playback. Tools continue unless cancelled.

Playback uses a latest-audio-response-wins policy: the first audio chunk from a
new response clears the previous response's queued audio and resampler state.
Late audio and completion events from superseded responses are ignored. Text-only
or tool-only responses do not replace playback. This switches speech when new
audio arrives, not immediately when a typed observation is submitted.

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
- `/reconnect`: open a new SDK session after a failure.
- `/quit` or Ctrl+C: abort the interaction and its tools.

The model calls `end_turn()` when the conversation ends. Final playback drains
before exit. Reconnection restores selected textual history as labeled context,
but waits for fresh speech or text instead of issuing inference against an empty audio
session. Audio is never replayed and completed tools are never re-executed.
Late results from an old session update context rather than using old call IDs.

## Tools And State

The demo registers current time and sandboxed `exec_command`. Set `EXA_API_KEY`
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
