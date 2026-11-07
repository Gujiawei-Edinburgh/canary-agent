"""Offline IPC fixture: requires host acknowledgments; never opens a socket."""
import base64
import json
import sys

from bridge_protocol import ready


def emit(event):
    print(json.dumps(event), flush=True)


config = json.loads(sys.stdin.readline())
emit(ready("fixture", 48000, playback_feedback=True, tool_calls=False, text_input=False))
context = {"run_id": "run", "turn_id": "turn"}
response_id = "response"
count = 0
for line in sys.stdin:
    command = json.loads(line)
    kind = command["type"]
    if kind == "audio":
        emit({"type": "response_started", "response_id": response_id})
        emit({"type": "audio_delta", "response_id": response_id, "item_id": "chunk",
              "chunk_id": "chunk", **context, "audio": base64.b64encode(bytes(4)).decode()})
        if config["model"] == "stop":
            emit({"type": "playback_stop", "response_id": response_id, **context})
            status = "interrupted"
        else:
            emit({"type": "playback_audio_done", **context})
            status = "completed"
        emit({"type": "response_finished", "response_id": response_id, "status": status})
    elif kind.startswith("playback_"):
        count += 1
        response = "ack-" + str(count)
        emit({"type": "response_started", "response_id": response})
        emit({"type": "text_delta", "response_id": response, "item_id": response,
              "text": json.dumps(command)})
        emit({"type": "response_finished", "response_id": response, "status": "completed"})
    elif kind == "close":
        break
