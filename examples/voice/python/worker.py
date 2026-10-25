"""Example-only JSONL bridge. DashScope owns WebSocket transport and decoding."""
import json
import logging
import sys
import threading


def sdk_conversation(config, callback):
    from dashscope.audio.qwen_omni import OmniRealtimeConversation
    return OmniRealtimeConversation(
        model=config["model"], url=config["endpoint"],
        api_key=config["api_key"], callback=callback,
    )


class Worker:
    def __init__(self, config, emit, factory=sdk_conversation):
        self.config = config
        self.emit = emit
        self.lock = threading.RLock()
        self.ready = threading.Event()
        self.closing = False
        self.ending = False
        self.active = set()
        self.interrupted = set()
        self.waiting = set()
        self.answered = set()
        self.items = {}
        self.arguments = {}
        self.text = {}
        self.speaking = False
        self.awaiting_vad = False
        self.requested = False
        self.needs_response = False
        self.has_input = False
        self.instructions = config.get("instructions", "")
        if config.get("history"):
            self.instructions += (
                "\nHistorical conversation data from the previous connection. "
                "Do not rerun completed tools. Accepted operations may still be running.\n"
                + json.dumps(config["history"], ensure_ascii=False)
            )
        self.conversation = factory(config, self)

    def configure(self):
        from dashscope.audio.qwen_omni import MultiModality
        self.conversation.update_session(
            output_modalities=[MultiModality.TEXT, MultiModality.AUDIO],
            voice=self.config["voice"],
            enable_turn_detection=True,
            enable_input_audio_transcription=True,
            input_audio_transcription_model="qwen3-asr-flash-realtime",
            instructions=self.instructions,
            tools=[{"type": "function", "function": tool}
                   for tool in self.config.get("tools", [])],
        )

    def start(self):
        self.conversation.connect()
        self.configure()
        if not self.ready.wait(20):
            raise TimeoutError("DashScope did not acknowledge session configuration")
        # A fresh Qwen connection has no audio input. Do not request inference
        # merely because historical text was supplied in its instructions.

    def on_open(self):
        pass

    def on_close(self, code, message):
        if not self.closing:
            self.emit({"type": "error", "message": f"DashScope connection closed: {code}: {message}"})

    def on_event(self, event):
        try:
            with self.lock:
                self.handle_event(event)
        except Exception as error:
            self.emit({"type": "error", "message": f"Qwen SDK event handling: {error}"})

    def handle_event(self, event):
        kind = event.get("type")
        if kind == "error" or (event.get("code") and event.get("message")):
            error = event.get("error", event)
            if error.get("message") == "Response request cancelled by new input":
                self.requested = False
                logging.info("Response request superseded by new input")
                return
            self.emit({"type": "error", "message": "DashScope: " + json.dumps(error, ensure_ascii=False)})
            return
        if kind == "session.updated":
            if not self.ready.is_set():
                self.ready.set()
                self.emit({"type": "ready"})
        elif kind == "input_audio_buffer.speech_started":
            self.speaking = True
            self.awaiting_vad = True
            self.emit({"type": "input_started", "item_id": event["item_id"]})
            for response in self.active - self.interrupted:
                self.interrupted.add(response)
                self.emit({"type": "response_finished", "response_id": response,
                           "status": "interrupted", "error": "new speech"})
        elif kind == "input_audio_buffer.speech_stopped":
            self.emit({"type": "input_ended", "item_id": event["item_id"]})
        elif kind == "input_audio_buffer.committed":
            self.speaking = False
            self.has_input = True
            self.emit({"type": "input_committed", "item_id": event["item_id"]})
        elif kind == "conversation.item.input_audio_transcription.completed":
            self.emit({"type": "transcript_finalized", "item_id": event["item_id"], "text": event["transcript"]})
        elif kind == "conversation.item.input_audio_transcription.failed":
            self.emit({"type": "error", "message": "Qwen transcription failed: " + json.dumps(event.get("error"))})
        elif kind == "response.created":
            response = event["response"]["id"]
            self.active.add(response)
            self.requested = False
            self.awaiting_vad = False
            self.needs_response = False
            self.emit({"type": "response_started", "response_id": response})
        elif kind in ("response.audio.delta", "response.output_audio.delta"):
            if event["response_id"] not in self.interrupted:
                self.emit({"type": "audio_delta", "response_id": event["response_id"],
                           "item_id": event["item_id"], "audio": event["delta"]})
        elif kind in ("response.audio_transcript.delta", "response.text.delta"):
            if event["response_id"] not in self.interrupted:
                key = (event["response_id"], event["item_id"])
                self.text[key] = self.text.get(key, "") + event["delta"]
                self.emit({"type": "text_delta", "response_id": key[0], "item_id": key[1], "text": event["delta"]})
        elif kind == "response.function_call_arguments.done":
            self.arguments[(event["response_id"], event["item_id"])] = event
        elif kind == "response.output_item.done":
            self.items.setdefault(event["response_id"], {})[event["item"]["id"]] = event["item"]
        elif kind == "response.done":
            self.finish_response(event)

    def finish_response(self, event):
        response = event["response"]
        response_id = response["id"]
        items = self.items.pop(response_id, {})
        for item in response.get("output", []):
            items[item["id"]] = item
        status = response.get("status")
        if status == "completed" and response_id not in self.interrupted:
            completed = []
            for item_id, item in items.items():
                calls = []
                if item["type"] == "function_call":
                    full = self.arguments.get((response_id, item_id), item)
                    args = json.loads(full["arguments"])
                    if not isinstance(args, dict):
                        raise ValueError("function arguments must be a JSON object")
                    calls.append({"call_id": full["call_id"], "name": full["name"], "arguments": args})
                text = "".join(p.get("transcript", p.get("text", "")) for p in item.get("content", []))
                text = text or self.text.get((response_id, item_id))
                completed.append({"type": "item_completed", "response_id": response_id,
                                  "item_id": item_id, "text": text, "function_calls": calls})
            for item in completed:
                self.waiting.update(call["call_id"] for call in item["function_calls"])
                self.emit(item)
        usage = response.get("usage")
        if usage:
            incoming, outgoing = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            cached = usage.get("input_tokens_details", usage.get("input_token_details", {})).get("cached_tokens", 0)
            self.emit({"type": "usage", "event_id": event.get("event_id", "usage_" + response_id),
                       "response_id": response_id, "usage": {"input_tokens": incoming,
                       "output_tokens": outgoing, "cached_input_tokens": cached,
                       "total_tokens": usage.get("total_tokens", incoming + outgoing)}})
        self.active.discard(response_id)
        if response_id not in self.interrupted:
            normalized = "completed" if status == "completed" else "interrupted" if status in ("cancelled", "canceled") else "failed"
            self.emit({"type": "response_finished", "response_id": response_id,
                       "status": normalized, "error": json.dumps(response.get("status_details"))})
        self.arguments = {key: value for key, value in self.arguments.items() if key[0] != response_id}
        self.text = {key: value for key, value in self.text.items() if key[0] != response_id}
        self.continue_if_idle()

    def continue_if_idle(self):
        if (self.needs_response and self.has_input and not self.active and not self.waiting
                and not self.speaking and not self.awaiting_vad and not self.requested and not self.ending):
            self.requested = True
            self.needs_response = False
            self.conversation.create_response()

    def command(self, command):
        with self.lock:
            kind = command["type"]
            if kind == "audio":
                self.conversation.append_audio(command["audio"])
            elif kind == "text":
                # Already journaled by Canary. Do not echo it as a transcript.
                self.conversation.create_item({
                    "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": command["text"]}],
                })
                self.has_input = True
                self.needs_response = True
                self.continue_if_idle()
            elif kind == "cancel":
                if command["response_id"] in self.active:
                    self.conversation.cancel_response()
            elif kind == "tool_result":
                call_id = command["call_id"]
                if call_id in self.answered:
                    return
                if call_id in self.waiting:
                    self.conversation.create_item({"type": "function_call_output", "call_id": call_id,
                                                   "output": json.dumps(command["result"], ensure_ascii=False)})
                    self.waiting.remove(call_id)
                    self.needs_response = True
                else:
                    # This operation belongs to a previous SDK connection.
                    self.instructions += "\nHistorical tool completion: " + json.dumps(command, ensure_ascii=False)
                    self.configure()
                self.answered.add(call_id)
                if command["name"] == "end_turn" and command["result"].get("output", {}).get("status") == "closed":
                    self.ending = True
                self.continue_if_idle()
            elif kind == "observation":
                self.instructions += "\nRuntime observation: " + json.dumps(command["event"], ensure_ascii=False)
                self.configure()
            elif kind == "close":
                self.closing = True
                self.conversation.close()
            else:
                raise ValueError("unknown bridge command: " + kind)


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    output_lock = threading.Lock()

    def emit(event):
        with output_lock:
            print(json.dumps(event, ensure_ascii=False), flush=True)

    worker = None
    try:
        config = json.loads(sys.stdin.readline())
        if config["type"] != "connect":
            raise ValueError("first bridge command must be connect")
        worker = Worker(config, emit)
        worker.start()
        for line in sys.stdin:
            command = json.loads(line)
            worker.command(command)
            if command["type"] == "close":
                break
    except Exception as error:
        emit({"type": "error", "message": f"Qwen SDK worker: {error}"})
    finally:
        if worker is not None:
            worker.closing = True
            worker.conversation.close()


if __name__ == "__main__":
    main()
