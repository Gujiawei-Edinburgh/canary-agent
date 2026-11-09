//! Application tools for observing and explicitly stopping local playback.
use crate::audio::PlaybackControl;
use canary_agent_runtime::{
    AgentError, FunctionContext, FunctionExecution, FunctionLimits, FunctionRecoveryPolicy,
    FunctionRegistry, FunctionSpec, SimpleFunction,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StopRequest {
    playback_id: String,
}

fn state_schema() -> Value {
    json!({"type":"object", "properties": {
        "playback_id":{"type":"string"}, "response_id":{"type":"string"},
        "status":{"type":"string", "enum":["queued","playing","completed","stopped"]},
        "received_audio_ms":{"type":"integer"}, "estimated_played_audio_ms":{"type":"integer"},
        "queued_audio_ms":{"type":"integer"}, "scheduled_audio_ms":{"type":"integer"},
        "generation_finished":{"type":"boolean"}, "generated_text_preview":{"type":"string"}
    }, "required":["playback_id","response_id","status","received_audio_ms","estimated_played_audio_ms","queued_audio_ms","scheduled_audio_ms","generation_finished","generated_text_preview"], "additionalProperties":false})
}

pub fn register_tools(registry: &mut FunctionRegistry, playback: PlaybackControl) {
    let observer = playback.clone();
    registry.register(SimpleFunction::new(
        FunctionSpec {
            name: "get_playback_state".into(),
            description: concat!(
                "Observe local speaker playback: IDs, queued audio, and estimated played cursor. ",
                "Use this before stop_playback to select an exact playback_id. Generation is not ",
                "playback, and the cursor does not identify exact words heard. This query does ",
                "not stop speech or request an interruption."
            ).into(),
            parameters: json!({"type":"object", "properties":{}, "additionalProperties":false}),
        },
        json!({"type":"object", "properties": {
            "playbacks":{"type":"array", "items":state_schema()},
            "cursor_is_estimate":{"type":"boolean"}, "scheduled_audio_cannot_be_recalled":{"type":"boolean"}
        }, "required":["playbacks","cursor_is_estimate","scheduled_audio_cannot_be_recalled"], "additionalProperties":false}),
        FunctionLimits { time_budget: Duration::from_secs(2), max_output_bytes: 1024 * 1024 },
        FunctionRecoveryPolicy::Idempotent,
        move |args: Value, context: FunctionContext| {
            let playback = observer.clone();
            async move {
                if context.abort_signal.is_cancelled() { return Err(AgentError::TurnExecution("playback query cancelled".into())); }
                if args.as_object().is_none_or(|args| !args.is_empty()) { return Err(AgentError::TurnExecution("get_playback_state expects an empty object".into())); }
                Ok(FunctionExecution::Completed { output: json!({"playbacks":playback.states()?, "cursor_is_estimate":true, "scheduled_audio_cannot_be_recalled":true}) })
            }
        },
    ));
    let mut nullable_state = state_schema();
    nullable_state["type"] = json!(["object", "null"]);
    registry.register(SimpleFunction::new(
        FunctionSpec {
            name: "stop_playback".into(),
            description: concat!(
                "Stop one specific local playback and discard only its queued audio. Obtain ",
                "playback_id from get_playback_state. Use when the user wants that speech to ",
                "stop or it is no longer appropriate, not merely because new audio or input ",
                "arrived. Other queued responses remain intact. Late audio for the stopped ID ",
                "is ignored. This does not cancel model generation or other tool operations."
            ).into(),
            parameters: json!({"type":"object", "properties":{"playback_id":{"type":"string","minLength":1}}, "required":["playback_id"], "additionalProperties":false}),
        },
        json!({"type":"object", "properties": {
            "status":{"type":"string", "enum":["stopped","already_stopped","already_completed","not_found"]},
            "playback_id":{"type":"string"}, "playback":nullable_state
        }, "required":["status","playback_id","playback"], "additionalProperties":false}),
        FunctionLimits { time_budget: Duration::from_secs(2), max_output_bytes: 1024 * 1024 },
        FunctionRecoveryPolicy::Idempotent,
        move |args: Value, context: FunctionContext| {
            let playback = playback.clone();
            async move {
                if context.abort_signal.is_cancelled() { return Err(AgentError::TurnExecution("playback stop cancelled".into())); }
                let request: StopRequest = serde_json::from_value(args)?;
                if request.playback_id.trim().is_empty() { return Err(AgentError::TurnExecution("playback_id must not be empty".into())); }
                Ok(FunctionExecution::Completed { output: serde_json::to_value(playback.stop(&request.playback_id)?)? })
            }
        },
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use canary_agent_kernel::ThreadProjection;
    use canary_agent_runtime::{AudioChunk, AudioEncoding, AudioFormat, turn_abort_pair};

    fn context(cancelled: bool) -> FunctionContext {
        let (abort, signal) = turn_abort_pair();
        if cancelled {
            abort.abort();
        }
        FunctionContext {
            thread_id: "thread".into(),
            turn_id: "turn".into(),
            call_id: "call".into(),
            metadata: json!({}),
            projection: ThreadProjection::default(),
            abort_signal: signal,
        }
    }

    #[tokio::test]
    async fn tools_expose_state_and_stop_only_the_requested_id() {
        let control = PlaybackControl::default();
        control.configure(48000);
        control.connect("connection");
        control.text_delta("response", "Introduction to Revolut");
        control
            .enqueue(
                "response",
                AudioChunk {
                    format: AudioFormat {
                        encoding: AudioEncoding::PcmS16Le,
                        sample_rate: 48000,
                        channels: 1,
                    },
                    data: vec![0; 9600],
                },
                None,
            )
            .unwrap();
        let mut registry = FunctionRegistry::new();
        register_tools(&mut registry, control.clone());
        let FunctionExecution::Completed { output } = registry
            .call("get_playback_state", json!({}), context(false))
            .await
            .unwrap()
        else {
            panic!("expected state")
        };
        assert_eq!(output["playbacks"][0]["playback_id"], "connection:response");
        assert_eq!(
            output["playbacks"][0]["generated_text_preview"],
            "Introduction to Revolut"
        );
        assert_eq!(output["playbacks"][0]["estimated_played_audio_ms"], 0);
        assert_eq!(output["cursor_is_estimate"], true);
        let FunctionExecution::Completed { output } = registry
            .call(
                "stop_playback",
                json!({"playback_id":"connection:response"}),
                context(false),
            )
            .await
            .unwrap()
        else {
            panic!("expected stop")
        };
        assert_eq!(output["status"], "stopped");
        assert_eq!(output["playback"]["status"], "stopped");
        assert!(
            registry
                .call(
                    "stop_playback",
                    json!({"playback_id":"connection:response"}),
                    context(true)
                )
                .await
                .is_err()
        );
        assert!(
            registry
                .call("stop_playback", json!({"playback_id":""}), context(false))
                .await
                .is_err()
        );
        assert!(
            registry
                .call("get_playback_state", json!({"extra":true}), context(false))
                .await
                .is_err()
        );
    }
}
