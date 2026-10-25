"""Offline process fixture for the Rust IPC test; does not import DashScope."""
import json
import sys


def emit(kind, **fields):
    print(json.dumps({"type": kind, **fields}), flush=True)


config = json.loads(sys.stdin.readline())
assert config["type"] == "connect"
emit("ready")
for line in sys.stdin:
    command = json.loads(line)
    if command["type"] == "audio":
        emit("input_started", item_id="user")
        emit("transcript_finalized", item_id="user", text="hello")
        emit("response_started", response_id="r")
        emit("item_completed", response_id="r", item_id="i", text=None,
             function_calls=[{"call_id": "call", "name": "get_current_time", "arguments": {}}])
        emit("response_finished", response_id="r", status="completed")
    elif command["type"] == "tool_result":
        assert command["call_id"] == "call"
        emit("response_started", response_id="r2")
        emit("audio_delta", response_id="r2", item_id="a", audio="AAA=")
        emit("response_finished", response_id="r2", status="completed")
    elif command["type"] == "text":
        assert command["text"] == "typed input"
        emit("response_started", response_id="text-response")
        emit("response_finished", response_id="text-response", status="completed")
    elif command["type"] == "close":
        break
