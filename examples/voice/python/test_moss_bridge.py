import base64
import json
import unittest

from moss_bridge import MossBridge, connection_url


class Socket:
    def __init__(self):
        self.sent = []

    async def send(self, packet):
        self.sent.append(packet)


class MossBridgeTests(unittest.IsolatedAsyncioTestCase):
    def test_connection_url_uses_generic_endpoint_and_model(self):
        self.assertEqual(connection_url({"endpoint": "wss://api.mosi.cn/v1", "model": "moss-live-1.0", "user": "a b"}),
                         "wss://api.mosi.cn/v1/realtime?model=moss-live-1.0&user=a+b")
        with self.assertRaises(ValueError):
            connection_url({"endpoint": "wss://example.com/v1?secret=value"})
    async def test_audio_context_and_host_driven_ack(self):
        ws, events = Socket(), []
        bridge = MossBridge(ws, events.append)
        await bridge.receive(json.dumps({"action": "response.audio.context", "data": {
            "chunk_id": "c", "run_id": "r", "turn_id": "t"}}))
        await bridge.receive(b"\x01\x02")
        self.assertEqual(events[0]["type"], "response_started")
        self.assertEqual(events[1]["chunk_id"], "c")
        self.assertEqual(base64.b64decode(events[1]["audio"]), b"\x01\x02")
        self.assertEqual(ws.sent, [])
        await bridge.command({"type": "playback_chunk_completed", "chunk_id": "c", "run_id": "r", "turn_id": "t"})
        self.assertEqual(json.loads(ws.sent[0])["action"], "playback.chunk.completed")

    async def test_requires_context_and_correct_input_length(self):
        bridge = MossBridge(Socket(), lambda _: None)
        with self.assertRaises(ValueError):
            await bridge.receive(b"audio")
        with self.assertRaises(ValueError):
            await bridge.command({"type": "audio", "audio": base64.b64encode(bytes(640)).decode()})
        bridge.ready = True
        with self.assertRaises(ValueError):
            await bridge.command({"type": "audio", "audio": base64.b64encode(b"odd").decode()})
        await bridge.command({"type": "audio", "audio": base64.b64encode(bytes(320)).decode()})
        self.assertEqual(bridge.ws.sent, [])
        await bridge.command({"type": "audio", "audio": base64.b64encode(bytes(320)).decode()})
        self.assertEqual(bridge.ws.sent[0], b"\x00" + bytes(640))

    async def test_reconnect_preserves_session_scope(self):
        ws = Socket()
        bridge = MossBridge(ws, lambda _: None)
        await bridge.start({"session_id": "previous", "memory_scope": "session"})
        self.assertEqual(json.loads(ws.sent[0]), {
            "action": "attach_session", "session_id": "previous", "memory_scope": "session"
        })

    async def test_interruption_is_not_acknowledged_automatically(self):
        ws, events = Socket(), []
        bridge = MossBridge(ws, events.append)
        await bridge.receive(json.dumps({"action": "response.audio.stop", "data": {"run_id": "r", "turn_id": "t"}}))
        self.assertEqual(ws.sent, [])
        await bridge.command({"type": "playback_stopped", "run_id": "r", "turn_id": "t", "played_audio_ms": 15})
        self.assertEqual(json.loads(ws.sent[0])["played_audio_ms"], 15)

    async def test_transcript_updates_are_replacements_and_end_is_normalized(self):
        events = []
        bridge = MossBridge(Socket(), events.append)
        await bridge.receive(json.dumps({"action": "output.transcript.updated", "data": {"run_id": "r", "turn_id": "t", "text": "hel"}}))
        await bridge.receive(json.dumps({"action": "output.transcript.done", "data": {"run_id": "r", "turn_id": "t", "text": "hello"}}))
        await bridge.receive(json.dumps({"action": "response.ended", "data": {"run_id": "r", "turn_id": "t", "status": "completed"}}))
        self.assertEqual([event["type"] for event in events], ["response_started", "text_delta", "item_completed", "response_finished"])
        self.assertEqual(events[2]["text"], "hello")

    async def test_generation_done_does_not_acknowledge_playback(self):
        ws, events = Socket(), []
        bridge = MossBridge(ws, events.append)
        data = {"run_id": "r", "turn_id": "t"}
        await bridge.receive(json.dumps({"action": "response.audio.done", "data": data}))
        self.assertEqual(events, [{"type": "playback_audio_done", **data}])
        self.assertEqual(ws.sent, [])
        await bridge.command({"type": "playback_completed", **data})
        self.assertEqual(json.loads(ws.sent[0]), {"action": "playback.completed", **data})

    async def test_finished_response_cannot_restart_from_late_audio(self):
        ws, events = Socket(), []
        bridge = MossBridge(ws, events.append)
        data = {"run_id": "r", "turn_id": "t"}
        await bridge.receive(json.dumps({"action": "response.interrupted", "data": data}))
        await bridge.receive(json.dumps({"action": "response.ended", "data": {**data, "status": "completed"}}))
        self.assertEqual(events[-1]["status"], "interrupted")
        count = len(events)
        await bridge.receive(json.dumps({"action": "response.audio.context", "data": {**data, "chunk_id": "late"}}))
        await bridge.receive(bytes(100))
        await bridge.receive(json.dumps({"action": "response.ended", "data": {**data, "status": "completed"}}))
        self.assertEqual(len(events), count)
