//! Local playback accounting. Device timestamps estimate delivery to the DAC,
//! not whether a person heard the sound. Never acknowledge network receipt.
use crate::bridge::{BridgeSender, PlaybackFeedback};
use canary_agent_runtime::{AudioChunk, Result};
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, mpsc};

pub type Writer = Arc<Mutex<BridgeSender>>;

#[derive(Clone)]
pub struct Target {
    pub connection_id: String,
    pub run_id: String,
    pub turn_id: String,
    pub writer: Writer,
}

impl Target {
    fn key(&self) -> (String, String, String) {
        (
            self.connection_id.clone(),
            self.run_id.clone(),
            self.turn_id.clone(),
        )
    }
    pub(crate) fn same_response(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.writer, &other.writer)
            && self.run_id == other.run_id
            && self.turn_id == other.turn_id
    }
}

pub struct Delivery {
    target: Target,
    feedback: PlaybackFeedback,
}

impl Delivery {
    pub async fn send(self) -> Result<()> {
        self.target.writer.lock().await.write(&self.feedback).await
    }
}

pub enum Event {
    Ready {
        connection_id: String,
        provider: String,
        tool_calls: bool,
        text_input: bool,
    },
    Audio {
        response_id: String,
        chunk_id: String,
        chunk: AudioChunk,
        target: Option<Target>,
    },
    TextDelta {
        response_id: String,
        text: String,
    },
    Stop {
        response_id: String,
        target: Target,
    },
    AudioDone {
        target: Target,
    },
    Finished {
        response_id: String,
    },
}

pub type Events = mpsc::UnboundedReceiver<Event>;

struct Chunk {
    id: String,
    start: u64,
    end: u64,
    spans: Vec<Span>,
}

struct Span {
    start: Instant,
    frames: u64,
    rate: u32,
}

impl Span {
    fn end(&self) -> Instant {
        self.start + Duration::from_secs_f64(self.frames as f64 / self.rate as f64)
    }
    fn played(&self, now: Instant) -> u64 {
        (now.saturating_duration_since(self.start).as_nanos() * u128::from(self.rate)
            / 1_000_000_000)
            .min(u128::from(self.frames)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge;
    use std::path::Path;

    fn target(connection: &str) -> Target {
        let (writer, _receiver) = bridge::spawn(
            "python3",
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("python/mock_worker.py"),
        )
        .unwrap();
        Target {
            connection_id: connection.into(),
            run_id: "run".into(),
            turn_id: "turn".into(),
            writer: Arc::new(Mutex::new(writer)),
        }
    }

    #[tokio::test]
    async fn acknowledges_only_after_device_delivery_and_generation_done() {
        let target = target("connection");
        let mut tracker = Tracker::default();
        let now = Instant::now();
        tracker.enqueue(target.clone(), "chunk".into(), 0, 4800);
        assert!(tracker.drain(now).is_empty());
        assert!(tracker.audio_done(target));
        assert!(tracker.drain(now).is_empty());
        let playback = now + Duration::from_millis(50);
        tracker.rendered(0, 4800, playback, 48000);
        assert_eq!(
            tracker.deadline(),
            Some(playback + Duration::from_millis(100))
        );
        assert!(
            tracker
                .drain(playback + Duration::from_millis(99))
                .is_empty()
        );
        let deliveries = tracker.drain(playback + Duration::from_millis(100));
        assert_eq!(deliveries.len(), 2);
        assert!(
            matches!(&deliveries[0].feedback, PlaybackFeedback::ChunkCompleted { chunk_id, .. } if chunk_id == "chunk")
        );
        assert!(matches!(
            deliveries[1].feedback,
            PlaybackFeedback::Completed { .. }
        ));
        assert!(tracker.drain(playback + Duration::from_secs(1)).is_empty());
    }

    #[tokio::test]
    async fn interruption_reports_only_current_unacknowledged_chunk() {
        let target = target("connection");
        let mut tracker = Tracker::default();
        let now = Instant::now();
        tracker.enqueue(target.clone(), "first".into(), 0, 4800);
        tracker.enqueue(target.clone(), "second".into(), 4800, 9600);
        tracker.rendered(0, 9600, now, 48000);
        tracker.stop(now + Duration::from_millis(150));
        let deliveries = tracker.drain(now + Duration::from_millis(150));
        assert_eq!(deliveries.len(), 2);
        assert!(matches!(
            deliveries[0].feedback,
            PlaybackFeedback::ChunkCompleted { .. }
        ));
        assert!(matches!(
            deliveries[1].feedback,
            PlaybackFeedback::Stopped {
                played_audio_ms: 50,
                ..
            }
        ));
        assert!(!tracker.audio_done(target.clone()));
        assert!(!tracker.stop_target(&target, now + Duration::from_secs(1)));
        assert!(tracker.drain(now + Duration::from_secs(1)).is_empty());
    }

    #[tokio::test]
    async fn scheduled_but_not_played_audio_is_not_acknowledged_on_stop() {
        let target = target("connection");
        let now = Instant::now();
        let mut tracker = Tracker::default();
        tracker.enqueue(target, "chunk".into(), 0, 4800);
        tracker.rendered(0, 4800, now + Duration::from_millis(50), 48000);
        tracker.stop(now);
        let deliveries = tracker.drain(now);
        assert_eq!(deliveries.len(), 1);
        assert!(matches!(
            deliveries[0].feedback,
            PlaybackFeedback::Stopped {
                played_audio_ms: 0,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn chunk_can_span_callbacks_and_underruns() {
        let target = target("connection");
        let now = Instant::now();
        let mut tracker = Tracker::default();
        tracker.enqueue(target, "chunk".into(), 10, 4810);
        tracker.rendered(0, 2410, now, 48000);
        assert!(tracker.deadline().is_none());
        assert!(tracker.drain(now + Duration::from_secs(1)).is_empty());
        tracker.rendered(2410, 4810, now + Duration::from_secs(1), 48000);
        assert!(tracker.drain(now + Duration::from_millis(1049)).is_empty());
        assert_eq!(tracker.drain(now + Duration::from_millis(1050)).len(), 1);
    }

    #[tokio::test]
    async fn reconnect_keeps_feedback_bound_to_original_writer() {
        let old = target("old");
        let new = target("new");
        let now = Instant::now();
        let mut tracker = Tracker::default();
        tracker.enqueue(old.clone(), "old-chunk".into(), 0, 4800);
        tracker.stop(now);
        tracker.enqueue(new.clone(), "new-chunk".into(), 0, 4800);
        let deliveries = tracker.drain(now);
        assert_eq!(deliveries.len(), 1);
        assert!(Arc::ptr_eq(&deliveries[0].target.writer, &old.writer));
        assert!(!Arc::ptr_eq(&deliveries[0].target.writer, &new.writer));
        assert!(!tracker.stop_target(&old, now));
        assert!(tracker.matches(&new));
        assert!(!tracker.audio_done(old));
        assert!(tracker.drain(now).is_empty());
    }

    #[tokio::test]
    async fn empty_audio_done_is_acknowledged_once() {
        let target = target("connection");
        let mut tracker = Tracker::default();
        assert!(!tracker.audio_done(target.clone()));
        assert!(!tracker.audio_done(target));
        let deliveries = tracker.drain(Instant::now());
        assert_eq!(deliveries.len(), 1);
        assert!(matches!(
            deliveries[0].feedback,
            PlaybackFeedback::Completed { .. }
        ));
    }
}

struct Response {
    target: Target,
    chunks: VecDeque<Chunk>,
    done: bool,
}

#[derive(Default)]
pub struct Tracker {
    response: Option<Response>,
    outbox: Vec<Delivery>,
    closed: VecDeque<(String, String, String)>,
}

impl Tracker {
    pub fn matches(&self, target: &Target) -> bool {
        self.response
            .as_ref()
            .is_some_and(|r| r.target.same_response(target))
    }
    pub fn enqueue(&mut self, target: Target, id: String, start: u64, end: u64) {
        if self.closed.contains(&target.key()) {
            return;
        }
        if !self
            .response
            .as_ref()
            .is_some_and(|r| r.target.same_response(&target))
        {
            self.stop(Instant::now());
            self.response = Some(Response {
                target,
                chunks: VecDeque::new(),
                done: false,
            });
        }
        self.response.as_mut().unwrap().chunks.push_back(Chunk {
            id,
            start,
            end,
            spans: Vec::new(),
        });
    }

    pub fn rendered(&mut self, start: u64, end: u64, playback: Instant, rate: u32) {
        if let Some(response) = &mut self.response {
            for chunk in &mut response.chunks {
                let from = start.max(chunk.start);
                let to = end.min(chunk.end);
                if from < to {
                    chunk.spans.push(Span {
                        start: playback
                            + Duration::from_secs_f64((from - start) as f64 / rate as f64),
                        frames: to - from,
                        rate,
                    });
                }
            }
        }
    }

    pub fn audio_done(&mut self, target: Target) -> bool {
        if self.closed.contains(&target.key()) {
            return false;
        }
        if self.response.is_none() || !self.matches(&target) {
            self.close(&target);
            self.outbox.push(Delivery {
                feedback: PlaybackFeedback::Completed {
                    run_id: target.run_id.clone(),
                    turn_id: target.turn_id.clone(),
                },
                target,
            });
        } else if let Some(response) = &mut self.response
            && response.target.same_response(&target)
            && !response.done
        {
            response.done = true;
            return true;
        }
        false
    }

    pub fn stop_target(&mut self, target: &Target, now: Instant) -> bool {
        if self.matches(target) {
            self.stop(now);
            return true;
        }
        if !self.closed.contains(&target.key()) {
            self.close(target);
            self.outbox.push(Delivery {
                target: target.clone(),
                feedback: PlaybackFeedback::Stopped {
                    run_id: target.run_id.clone(),
                    turn_id: target.turn_id.clone(),
                    played_audio_ms: 0,
                },
            });
        }
        false
    }

    pub fn stop(&mut self, now: Instant) {
        self.poll(now);
        if let Some(response) = self.response.take() {
            self.close(&response.target);
            // The protocol asks for progress in the first unacknowledged chunk,
            // not cumulative playback across chunks already acknowledged.
            let played_audio_ms = response.chunks.front().map_or(0, |chunk| {
                (chunk
                    .spans
                    .iter()
                    .map(|span| span.played(now) as f64 / f64::from(span.rate))
                    .sum::<f64>()
                    * 1000.0)
                    .floor() as u64
            });
            self.outbox.push(Delivery {
                feedback: PlaybackFeedback::Stopped {
                    run_id: response.target.run_id.clone(),
                    turn_id: response.target.turn_id.clone(),
                    played_audio_ms,
                },
                target: response.target,
            });
        }
    }

    pub fn poll(&mut self, now: Instant) {
        let Some(response) = &mut self.response else {
            return;
        };
        while response.chunks.front().is_some_and(|chunk| {
            chunk.spans.iter().map(|s| s.frames).sum::<u64>() == chunk.end - chunk.start
                && chunk.spans.last().is_none_or(|span| span.end() <= now)
        }) {
            let chunk = response.chunks.pop_front().unwrap();
            self.outbox.push(Delivery {
                target: response.target.clone(),
                feedback: PlaybackFeedback::ChunkCompleted {
                    chunk_id: chunk.id,
                    run_id: response.target.run_id.clone(),
                    turn_id: response.target.turn_id.clone(),
                },
            });
        }
        if response.done && response.chunks.is_empty() {
            let response = self.response.take().unwrap();
            self.close(&response.target);
            self.outbox.push(Delivery {
                feedback: PlaybackFeedback::Completed {
                    run_id: response.target.run_id.clone(),
                    turn_id: response.target.turn_id.clone(),
                },
                target: response.target,
            });
        }
    }

    fn close(&mut self, target: &Target) {
        self.closed.push_back(target.key());
        if self.closed.len() > 256 {
            self.closed.pop_front();
        }
    }

    pub fn deadline(&self) -> Option<Instant> {
        let chunk = self.response.as_ref()?.chunks.front()?;
        (chunk.spans.iter().map(|s| s.frames).sum::<u64>() == chunk.end - chunk.start)
            .then(|| chunk.spans.last().map(|s| s.end()))
            .flatten()
    }

    pub fn drain(&mut self, now: Instant) -> Vec<Delivery> {
        self.poll(now);
        std::mem::take(&mut self.outbox)
    }
}
