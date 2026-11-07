"""Experimental MOSS transport. Playback acknowledgments are host-driven."""
import asyncio
import base64
import json
import sys
import uuid
from urllib.parse import urlencode, urlsplit, urlunsplit

from bridge_protocol import ready


class MossBridge:
    def __init__(self, ws, emit):
        self.ws = ws
        self.emit = emit
        self.audio_context = None
        self.audio_input = bytearray()
        self.ready = False
        self.responses = set()
        self.transcripts = {}
        self.finished = set()
        self.interrupted = set()

    @staticmethod
    def response_id(data):
        return json.dumps([data["run_id"], data["turn_id"]], separators=(",", ":"))

    def start_response(self, data):
        response_id = self.response_id(data)
        if response_id not in self.responses:
            self.responses.add(response_id)
            self.emit({"type": "response_started", "response_id": response_id})
        return response_id

    async def start(self, config):
        attach = {"action": "attach_session"}
        if config.get("session_id"):
            attach["session_id"] = config["session_id"]
        if config.get("memory_scope"):
            attach["memory_scope"] = config["memory_scope"]
        await self.ws.send(json.dumps(attach))

    async def command(self, command):
        kind = command.get("type")
        if kind == "audio":
            if not self.ready:
                raise ValueError("MOSS session is not ready for audio")
            audio = base64.b64decode(command["audio"], validate=True)
            if len(audio) % 2:
                raise ValueError("MOSS input must be aligned 16 kHz mono PCM16")
            self.audio_input.extend(audio)
            while len(self.audio_input) >= 640:
                frame = bytes(self.audio_input[:640])
                del self.audio_input[:640]
                await self.ws.send(b"\x00" + frame)
        elif kind == "playback_chunk_completed":
            await self.ws.send(json.dumps({"action": "playback.chunk.completed", **{
                key: command[key] for key in ("chunk_id", "run_id", "turn_id")
            }}))
        elif kind == "playback_completed":
            await self.ws.send(json.dumps({"action": "playback.completed", **{
                key: command[key] for key in ("run_id", "turn_id")
            }}))
        elif kind == "playback_stopped":
            await self.ws.send(json.dumps({"action": "playback.stopped", **{
                key: command[key] for key in ("run_id", "turn_id", "played_audio_ms")
            }}))
        elif kind == "close":
            await self.ws.close()
        else:
            raise ValueError(f"unsupported MOSS command: {kind}")

    async def receive(self, packet):
        if isinstance(packet, bytes):
            if self.audio_context is None:
                raise ValueError("MOSS audio arrived without context")
            context, self.audio_context = self.audio_context, None
            if self.response_id(context) in self.finished:
                return
            response_id = self.start_response(context)
            self.emit({"type": "audio_delta", **context,
                       "response_id": response_id,
                       "item_id": context["chunk_id"],
                       "audio": base64.b64encode(packet).decode("ascii")})
            return
        event = json.loads(packet)
        action = event.get("action")
        data = event.get("data") or {}
        if action == "response.audio.context":
            if self.audio_context is not None:
                raise ValueError("MOSS audio context arrived before previous audio")
            self.audio_context = {key: data[key] for key in ("chunk_id", "run_id", "turn_id")}
        elif action == "input.transcript.done":
            item_id = "moss_input_" + uuid.uuid4().hex
            self.emit({"type": "input_committed", "item_id": item_id})
            self.emit({"type": "transcript_finalized", "item_id": item_id, "text": data["text"]})
        elif action in ("output.transcript.updated", "output.transcript.done"):
            if self.response_id(data) in self.finished:
                return
            response_id = self.start_response(data)
            self.transcripts[response_id] = data["text"]
        elif action == "response.interrupted":
            response_id = self.response_id(data)
            self.interrupted.add(response_id)
            if data.get("text"):
                self.transcripts[response_id] = data["text"]
        elif action == "response.ended":
            if self.response_id(data) in self.finished:
                return
            response_id = self.start_response(data)
            text = self.transcripts.pop(response_id, None)
            if text:
                self.emit({"type": "text_delta", "response_id": response_id,
                           "item_id": response_id + ":text", "text": text})
                self.emit({"type": "item_completed", "response_id": response_id,
                           "item_id": response_id + ":text", "text": text, "function_calls": []})
            status = data["status"]
            if response_id in self.interrupted:
                status = "aborted"
                self.interrupted.discard(response_id)
            self.emit({"type": "response_finished", "response_id": response_id,
                       "status": {"completed": "completed", "aborted": "interrupted"}.get(status, "failed"),
                       "error": "" if status == "completed" else status})
            self.responses.discard(response_id)
            self.finished.add(response_id)
        elif action == "response.audio.stop":
            self.emit({"type": "playback_stop", "response_id": self.response_id(data),
                       "run_id": data["run_id"], "turn_id": data["turn_id"]})
        elif action == "response.audio.done":
            self.emit({"type": "playback_audio_done", "run_id": data["run_id"], "turn_id": data["turn_id"]})
        elif action == "error":
            self.emit({"type": "error", "message": str(data.get("error", "MOSS provider error"))})


async def run(config, emit, stdin):
    import websockets

    url = connection_url(config)
    async with websockets.connect(url, additional_headers={
        "Authorization": "Bearer " + config["api_key"]
    }) as ws:
        bridge = MossBridge(ws, emit)
        await bridge.start(config)

        async def incoming():
            async for packet in ws:
                if isinstance(packet, str):
                    event = json.loads(packet)
                    if event.get("action") == "session_attached":
                        await ws.send(json.dumps({"action": "session.update", "session": {
                            "instructions": config.get("instructions", "")
                        }}))
                    elif event.get("action") == "session.updated":
                        if (event.get("data") or {}).get("error"):
                            raise ValueError("MOSS session configuration failed")
                        await ws.send(json.dumps({"action": "media.mode", "binary_protocol": "tagged-v1"}))
                        await ws.send(json.dumps({"action": "mic_state", "enabled": True}))
                        bridge.ready = True
                        emit(ready("moss", 48000, playback_feedback=True,
                                   tool_calls=False, text_input=False))
                await bridge.receive(packet)

        async def outgoing():
            while True:
                line = await stdin.readline()
                if not line:
                    await ws.close()
                    return
                command = json.loads(line)
                await bridge.command(command)
                if command.get("type") == "close":
                    return

        receive_task = asyncio.create_task(incoming())
        send_task = asyncio.create_task(outgoing())
        done, pending = await asyncio.wait({receive_task, send_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()


def connection_url(config):
    endpoint = config.get("endpoint", "wss://api.mosi.cn/v1/realtime")
    parts = urlsplit(endpoint)
    if parts.scheme not in ("ws", "wss") or not parts.hostname or parts.query or parts.fragment:
        raise ValueError("invalid MOSS endpoint")
    path = parts.path.rstrip("/")
    if path in ("", "/v1"):
        path = "/v1/realtime"
    query = {"model": config.get("model", "moss-live-1.0")}
    if config.get("user"):
        query["user"] = config["user"]
    return urlunsplit((parts.scheme, parts.netloc, path, urlencode(query), ""))


def main():
    def emit(event):
        print(json.dumps(event, ensure_ascii=False), flush=True)

    async def start():
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)
        try:
            config = json.loads(await reader.readline())
            if config.get("type") != "connect":
                raise ValueError("first command must be connect")
            await run(config, emit, reader)
        finally:
            transport.close()

    try:
        asyncio.run(start())
    except Exception as error:
        emit({"type": "error", "message": str(error)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
