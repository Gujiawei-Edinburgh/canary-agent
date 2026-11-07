"""Shared JSONL bridge protocol metadata for realtime providers."""


def ready(provider, output_rate, *, playback_feedback, tool_calls, text_input):
    return {
        "type": "ready",
        "provider": provider,
        "audio": {"input_rate": 16000, "output_rate": output_rate,
                  "channels": 1, "encoding": "pcm_s16le"},
        "capabilities": {"playback_feedback": playback_feedback,
                         "tool_calls": tool_calls, "text_input": text_input},
    }
