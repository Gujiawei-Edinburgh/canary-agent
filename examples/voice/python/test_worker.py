import sys
import types
import unittest
from unittest.mock import patch

from worker import Worker


class FakeConversation:
    def __init__(self, config, callback):
        self.callback = callback
        self.calls = []

    def connect(self):
        self.calls.append("connect")

    def update_session(self, **config):
        self.calls.append({"update": config})
        self.callback.on_event({"type": "session.updated"})

    def append_audio(self, audio):
        self.calls.append({"audio": audio})

    def create_item(self, item):
        self.calls.append({"item": item})

    def create_response(self):
        self.calls.append("response")

    def cancel_response(self):
        self.calls.append("cancel")

    def close(self):
        self.calls.append("close")


class WorkerTests(unittest.TestCase):
    def setUp(self):
        sdk = types.ModuleType("dashscope.audio.qwen_omni")
        sdk.MultiModality = types.SimpleNamespace(TEXT="text", AUDIO="audio")
        self.modules = patch.dict(sys.modules, {"dashscope": types.ModuleType("dashscope"),
            "dashscope.audio": types.ModuleType("dashscope.audio"),
            "dashscope.audio.qwen_omni": sdk})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.events = []
        self.worker = Worker({"model": "test", "voice": "Tina", "history": [{"past": "data"}]},
                             self.events.append, FakeConversation)
        self.worker.start()
        self.sdk = self.worker.conversation

    def event(self, kind, **fields):
        self.worker.on_event({"type": kind, **fields})

    def response(self):
        self.event("input_audio_buffer.speech_started", item_id="user")
        self.event("input_audio_buffer.committed", item_id="user")
        self.event("response.created", response={"id": "response"})

    def test_start_restores_context_but_does_not_request_inference(self):
        self.assertEqual(self.events, [{"type": "ready"}])
        self.assertNotIn("response", self.sdk.calls)
        self.assertIn("Historical conversation", self.sdk.calls[1]["update"]["instructions"])

    def test_text_starts_response_without_audio_or_duplicate_transcript(self):
        self.worker.command({"type": "text", "text": "hello"})
        self.assertEqual(self.sdk.calls[-2:], [
            {"item": {"type": "message", "role": "user",
                      "content": [{"type": "input_text", "text": "hello"}]}},
            "response",
        ])
        self.assertEqual(self.events, [{"type": "ready"}])

    def test_text_during_response_is_delivered_without_cancellation(self):
        self.response()
        self.worker.command({"type": "text", "text": "also check the time"})
        self.assertEqual(self.sdk.calls[-1]["item"]["content"][0]["text"], "also check the time")
        self.assertNotIn("cancel", self.sdk.calls)
        self.assertNotIn("response", self.sdk.calls)
        self.event("response.done", response={"id": "response", "status": "completed"})
        self.assertEqual(self.sdk.calls.count("response"), 1)

    def test_text_during_tool_wait_is_delivered_and_result_continues(self):
        self.worker.waiting.add("call")
        self.worker.command({"type": "text", "text": "new requirement"})
        self.assertIn("item", self.sdk.calls[-1])
        self.assertNotIn("response", self.sdk.calls)
        self.worker.command({"type": "tool_result", "call_id": "call", "name": "test",
                             "result": {"output": {}}})
        self.assertEqual(self.sdk.calls.count("response"), 1)

    def test_text_during_speech_leaves_response_to_vad(self):
        self.event("input_audio_buffer.speech_started", item_id="speech")
        self.worker.command({"type": "text", "text": "context"})
        self.assertIn("item", self.sdk.calls[-1])
        self.assertNotIn("response", self.sdk.calls)
        self.event("input_audio_buffer.committed", item_id="speech")
        self.event("response.created", response={"id": "r"})
        self.event("response.done", response={"id": "r", "status": "completed"})
        self.assertNotIn("response", self.sdk.calls)

    def test_tools_wait_for_response_done_and_result_continues_once(self):
        self.response()
        self.event("response.function_call_arguments.done", response_id="response", item_id="tool",
                   call_id="call", name="exec_command", arguments='{"cmd":"ls"}')
        item = {"id": "tool", "type": "function_call", "call_id": "call", "name": "exec_command", "arguments": '{"cmd":"truncated'}
        self.event("response.output_item.done", response_id="response", item=item)
        self.assertFalse(any(e["type"] == "item_completed" for e in self.events))
        self.event("response.done", response={"id": "response", "status": "completed", "output": [item]})
        call = next(e for e in self.events if e["type"] == "item_completed")["function_calls"][0]
        self.assertEqual(call["arguments"], {"cmd": "ls"})
        result = {"type": "tool_result", "call_id": "call", "name": "exec_command", "result": {"output": {"stdout": "files"}}}
        self.worker.command(result)
        self.worker.command(result)
        self.assertEqual(self.sdk.calls.count("response"), 1)
        self.assertEqual(sum("item" in call for call in self.sdk.calls if isinstance(call, dict)), 1)

    def test_interruption_discards_calls_and_preserves_usage(self):
        self.response()
        self.event("input_audio_buffer.speech_started", item_id="interrupt")
        self.event("response.output_item.done", response_id="response", item={"id": "tool", "type": "function_call", "arguments": "{"})
        self.event("response.done", response={"id": "response", "status": "cancelled", "usage": {"input_tokens": 5, "output_tokens": 2}})
        self.assertFalse(any(e["type"] == "item_completed" for e in self.events))
        self.assertEqual(sum(e["type"] == "response_finished" for e in self.events), 1)
        self.assertEqual(next(e for e in self.events if e["type"] == "usage")["usage"]["total_tokens"], 7)

    def test_old_result_updates_context_without_reusing_call_id(self):
        self.worker.command({"type": "tool_result", "call_id": "old", "name": "exec_command", "result": {"output": {"stdout": "files"}}})
        self.assertNotIn("response", self.sdk.calls)
        self.assertFalse(any("item" in call for call in self.sdk.calls if isinstance(call, dict)))
        self.assertEqual(sum(e["type"] == "ready" for e in self.events), 1)


if __name__ == "__main__":
    unittest.main()
