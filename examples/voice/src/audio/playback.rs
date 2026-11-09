//! FIFO playback with independent, connection-scoped response buffers.
use super::{Converter, audio_error};
use crate::feedback::{Delivery, Target, Tracker};
use canary_agent_runtime::{AudioChunk, AudioEncoding, Result};
use rubato::Resampler;
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::Notify;

pub struct Tracking {
    pub chunk_id: String,
    pub target: Target,
}

#[derive(Clone, Serialize)]
pub struct PlaybackState {
    pub playback_id: String,
    pub response_id: String,
    pub status: &'static str,
    pub received_audio_ms: u64,
    pub estimated_played_audio_ms: u64,
    pub queued_audio_ms: u64,
    pub scheduled_audio_ms: u64,
    pub generation_finished: bool,
    pub generated_text_preview: String,
}

#[derive(Serialize)]
pub struct StopResult {
    pub status: &'static str,
    pub playback_id: String,
    pub playback: Option<PlaybackState>,
}

struct Span {
    start: Instant,
    frames: u64,
    rate: u32,
}

impl Span {
    fn played(&self, now: Instant) -> u64 {
        (now.saturating_duration_since(self.start).as_nanos() * u128::from(self.rate)
            / 1_000_000_000)
            .min(u128::from(self.frames)) as u64
    }
    fn end(&self) -> Instant {
        self.start + Duration::from_secs_f64(self.frames as f64 / f64::from(self.rate))
    }
}

struct Response {
    id: String,
    response_id: String,
    input_rate: u32,
    output_rate: u32,
    converter: Option<Converter>,
    input_frames: u64,
    raw_frames: u64,
    position: u64,
    samples: VecDeque<f32>,
    spans: Vec<Span>,
    played_base: u64,
    finished: bool,
    stopped: bool,
    target: Option<Target>,
    tracker: Tracker,
    text_preview: String,
}

impl Response {
    fn new(id: String, response_id: &str, input_rate: u32, output_rate: u32) -> Result<Self> {
        Ok(Self {
            id,
            response_id: response_id.into(),
            input_rate,
            output_rate,
            converter: Some(Converter::new(input_rate, output_rate)?),
            input_frames: 0,
            raw_frames: 0,
            position: 0,
            samples: VecDeque::new(),
            spans: Vec::new(),
            finished: false,
            stopped: false,
            played_base: 0,
            target: None,
            tracker: Tracker::default(),
            text_preview: String::new(),
        })
    }

    fn data_end(&self) -> u64 {
        self.input_frames * u64::from(self.output_rate) / u64::from(self.input_rate)
    }

    fn append_output(&mut self, output: Vec<f32>, delay: u64) {
        // Remove filter warm-up and flush padding, not real audio. Consecutive
        // responses must not acquire an artificial silence gap at each boundary.
        let start = delay
            .saturating_sub(self.raw_frames)
            .min(output.len() as u64) as usize;
        let end = (delay + self.data_end())
            .saturating_sub(self.raw_frames)
            .min(output.len() as u64) as usize;
        self.raw_frames += output.len() as u64;
        if start < end {
            self.samples.extend(output[start..end].iter().copied());
        }
    }

    fn enqueue(&mut self, chunk: AudioChunk, tracking: Option<Tracking>) -> Result<()> {
        if self.stopped || self.finished {
            return Ok(());
        }
        if chunk.format.sample_rate != self.input_rate {
            return Err(audio_error("playback response changed sample rate"));
        }
        let start = self.data_end();
        self.input_frames += (chunk.data.len() / 2) as u64;
        let end = self.data_end();
        let converter = self.converter.as_mut().expect("unfinished converter");
        let output = converter.push(
            chunk
                .data
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0),
        )?;
        let delay = converter.resampler.output_delay() as u64;
        if let Some(tracking) = tracking {
            self.target = Some(tracking.target.clone());
            self.tracker
                .enqueue(tracking.target, tracking.chunk_id, start, end);
        }
        self.append_output(output, delay);
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.stopped || self.finished {
            return Ok(());
        }
        let mut converter = self.converter.take().expect("unfinished converter");
        let padding = converter.resampler.input_frames_next() * 2;
        let delay = converter.resampler.output_delay() as u64;
        let output = converter.push(vec![0.0; padding])?;
        self.append_output(output, delay);
        self.finished = true;
        Ok(())
    }

    fn played(&self, now: Instant) -> u64 {
        self.played_base + self.spans.iter().map(|s| s.played(now)).sum::<u64>()
    }

    fn retire_spans(&mut self, now: Instant) {
        let count = self
            .spans
            .iter()
            .take_while(|span| span.end() <= now)
            .count();
        self.played_base += self
            .spans
            .drain(..count)
            .map(|span| span.frames)
            .sum::<u64>();
    }

    fn state(&self, now: Instant) -> PlaybackState {
        let played = self.played(now).min(self.data_end());
        let ms = |frames: u64| frames * 1000 / u64::from(self.output_rate);
        PlaybackState {
            playback_id: self.id.clone(),
            response_id: self.response_id.clone(),
            status: if self.stopped {
                "stopped"
            } else if self.finished && played >= self.data_end() {
                "completed"
            } else if self.position > 0 {
                "playing"
            } else {
                "queued"
            },
            received_audio_ms: self.input_frames * 1000 / u64::from(self.input_rate),
            estimated_played_audio_ms: ms(played),
            queued_audio_ms: if self.stopped {
                0
            } else {
                ms(self.data_end().saturating_sub(self.position))
            },
            scheduled_audio_ms: ms(self.position.saturating_sub(played)),
            generation_finished: self.finished,
            generated_text_preview: self.text_preview.clone(),
        }
    }

    fn stop(&mut self, now: Instant) {
        if self.stopped {
            return;
        }
        self.tracker.stop(now);
        self.samples.clear();
        self.converter = None;
        self.stopped = true;
    }
}

#[derive(Default)]
struct Buffer {
    connection: String,
    rate: u32,
    order: VecDeque<String>,
    responses: BTreeMap<String, Response>,
    orphan_feedback: Tracker,
    closed: HashSet<String>,
    text_previews: BTreeMap<String, String>,
}

#[derive(Clone, Default)]
pub struct PlaybackControl {
    buffer: Arc<Mutex<Buffer>>,
    pub changed: Arc<Notify>,
    pub drained: Arc<Notify>,
}

impl PlaybackControl {
    pub fn configure(&self, rate: u32) {
        self.buffer.lock().expect("playback buffer").rate = rate;
    }

    pub fn connect(&self, connection: &str) {
        self.stop_all();
        let mut buffer = self.buffer.lock().expect("playback buffer");
        buffer.connection = connection.into();
        buffer.text_previews.clear();
    }

    pub fn enqueue(
        &self,
        response_id: &str,
        chunk: AudioChunk,
        tracking: Option<Tracking>,
    ) -> Result<()> {
        if chunk.format.encoding != AudioEncoding::PcmS16Le
            || chunk.format.channels != 1
            || chunk.format.sample_rate == 0
            || !chunk.data.len().is_multiple_of(2)
        {
            return Err(audio_error("expected mono PCM16 playback"));
        }
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| audio_error("playback queue poisoned"))?;
        if buffer.rate == 0 {
            return Err(audio_error("speaker not initialized"));
        }
        let id = format!("{}:{response_id}", buffer.connection);
        if buffer.closed.contains(&id) {
            return Ok(());
        }
        if !buffer.responses.contains_key(&id) {
            let mut response = Response::new(
                id.clone(),
                response_id,
                chunk.format.sample_rate,
                buffer.rate,
            )?;
            response.text_preview = buffer.text_previews.remove(&id).unwrap_or_default();
            buffer.responses.insert(id.clone(), response);
            buffer.order.push_back(id.clone());
        }
        // Keep the existing example's five-minute bound. This is a hard buffer
        // error, never an implicit decision to discard an older response.
        let queued: usize = buffer.responses.values().map(|r| r.samples.len()).sum();
        let incoming = (chunk.data.len() / 2) as u64 * u64::from(buffer.rate)
            / u64::from(chunk.format.sample_rate);
        if queued as u64 + incoming > u64::from(buffer.rate) * 300 {
            return Err(audio_error(
                "playback exceeded the example's 300-second buffer",
            ));
        }
        buffer
            .responses
            .get_mut(&id)
            .unwrap()
            .enqueue(chunk, tracking)?;
        Ok(())
    }

    pub fn finish(&self, response_id: &str) -> Result<()> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| audio_error("playback queue poisoned"))?;
        let id = format!("{}:{response_id}", buffer.connection);
        if let Some(response) = buffer.responses.get_mut(&id) {
            response.finish()?;
        }
        buffer.text_previews.remove(&id);
        buffer.closed.insert(id);
        self.changed.notify_one();
        Ok(())
    }

    pub fn text_delta(&self, response_id: &str, text: &str) {
        let mut buffer = self.buffer.lock().expect("playback buffer");
        let id = format!("{}:{response_id}", buffer.connection);
        let preview = if let Some(response) = buffer.responses.get_mut(&id) {
            &mut response.text_preview
        } else {
            if buffer.closed.contains(&id) {
                return;
            }
            buffer.text_previews.entry(id).or_default()
        };
        let remaining = 256usize.saturating_sub(preview.chars().count());
        preview.extend(text.chars().take(remaining));
    }

    pub fn audio_done(&self, target: Target) -> Result<()> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| audio_error("playback queue poisoned"))?;
        if let Some(response) = buffer
            .responses
            .values_mut()
            .find(|r| r.target.as_ref().is_some_and(|t| t.same_response(&target)))
        {
            response.finish()?;
            response.tracker.audio_done(target);
        } else {
            buffer.orphan_feedback.audio_done(target);
        }
        self.changed.notify_one();
        Ok(())
    }

    pub fn stop_target(&self, response_id: &str, target: &Target) {
        let mut buffer = self.buffer.lock().expect("playback buffer");
        buffer
            .closed
            .insert(format!("{}:{response_id}", target.connection_id));
        if let Some(response) = buffer
            .responses
            .values_mut()
            .find(|r| r.target.as_ref().is_some_and(|t| t.same_response(target)))
        {
            response.stop(Instant::now());
        } else {
            buffer.orphan_feedback.stop_target(target, Instant::now());
        }
        self.changed.notify_one();
        self.drained.notify_one();
    }

    pub fn states(&self) -> Result<Vec<PlaybackState>> {
        let buffer = self
            .buffer
            .lock()
            .map_err(|_| audio_error("playback queue poisoned"))?;
        let now = Instant::now();
        // Queue order is preserved; terminal history follows it for idempotent
        // stop results and late-chunk rejection.
        let mut states: Vec<_> = buffer
            .order
            .iter()
            .filter_map(|id| buffer.responses.get(id))
            .map(|r| r.state(now))
            .collect();
        states.extend(
            buffer
                .responses
                .values()
                .filter(|r| !buffer.order.contains(&r.id))
                .map(|r| r.state(now)),
        );
        Ok(states)
    }

    pub fn stop(&self, id: &str) -> Result<StopResult> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| audio_error("playback queue poisoned"))?;
        let now = Instant::now();
        let Some(response) = buffer.responses.get_mut(id) else {
            return Ok(StopResult {
                status: "not_found",
                playback_id: id.into(),
                playback: None,
            });
        };
        let status = match response.state(now).status {
            "stopped" => "already_stopped",
            "completed" => "already_completed",
            _ => {
                response.stop(now);
                "stopped"
            }
        };
        let result = StopResult {
            status,
            playback_id: id.into(),
            playback: Some(response.state(now)),
        };
        buffer.closed.insert(id.into());
        self.changed.notify_one();
        self.drained.notify_one();
        Ok(result)
    }

    pub fn stop_all(&self) {
        let mut buffer = self.buffer.lock().expect("playback buffer");
        let now = Instant::now();
        for response in buffer.responses.values_mut() {
            if response.state(now).status != "completed" {
                response.stop(now);
            }
        }
        let closed: Vec<_> = buffer.responses.keys().cloned().collect();
        buffer.closed.extend(closed);
        buffer.order.clear();
        self.changed.notify_one();
        self.drained.notify_one();
    }

    pub fn render(&self, frames: usize, playback: Instant) -> Vec<f32> {
        let mut output = vec![0.0; frames];
        let Ok(mut buffer) = self.buffer.lock() else {
            return output;
        };
        let rate = buffer.rate;
        let mut offset = 0;
        while offset < frames {
            let Some(id) = buffer.order.front().cloned() else {
                break;
            };
            let response = buffer.responses.get_mut(&id).expect("queued response");
            if response.stopped || response.samples.is_empty() && response.finished {
                buffer.order.pop_front();
                continue;
            }
            if response.samples.is_empty() {
                break;
            }
            let count = (frames - offset).min(response.samples.len());
            for sample in &mut output[offset..offset + count] {
                *sample = response.samples.pop_front().unwrap().clamp(-1.0, 1.0);
            }
            let at = playback + Duration::from_secs_f64(offset as f64 / f64::from(rate));
            let start = response.position;
            response.position += count as u64;
            response.spans.push(Span {
                start: at,
                frames: count as u64,
                rate,
            });
            response
                .tracker
                .rendered(start, response.position, at, rate);
            offset += count;
        }
        if offset > 0 {
            self.changed.notify_one();
            self.drained.notify_one();
        }
        output
    }

    pub fn feedback(&self) -> Vec<Delivery> {
        let mut buffer = self.buffer.lock().expect("playback buffer");
        let now = Instant::now();
        let mut outbox = Vec::new();
        for response in buffer.responses.values_mut() {
            outbox.extend(response.tracker.drain(now));
            response.retire_spans(now);
        }
        outbox.extend(buffer.orphan_feedback.drain(now));
        // Keep bounded terminal history while retaining ID tombstones so late
        // chunks cannot resurrect stopped or completed playback.
        let terminal: Vec<_> = buffer
            .responses
            .iter()
            .filter(|(_, r)| {
                matches!(r.state(now).status, "completed" | "stopped") && r.spans.is_empty()
            })
            .map(|(id, _)| id.clone())
            .collect();
        let excess = terminal.len().saturating_sub(32);
        for id in terminal.into_iter().take(excess) {
            buffer.responses.remove(&id);
            buffer.order.retain(|queued| queued != &id);
        }
        outbox
    }

    pub fn deadline(&self) -> Option<Instant> {
        self.buffer
            .lock()
            .ok()?
            .responses
            .values()
            .filter_map(|r| r.tracker.deadline())
            .min()
    }

    pub async fn wait_drained(&self) {
        loop {
            let notified = self.drained.notified();
            let deadline = {
                let buffer = self.buffer.lock().expect("playback buffer");
                if buffer.responses.values().any(|r| !r.samples.is_empty()) {
                    None
                } else {
                    let deadline = buffer
                        .responses
                        .values()
                        .flat_map(|r| r.spans.last().map(Span::end))
                        .max();
                    if deadline.is_none_or(|at| at <= Instant::now()) {
                        return;
                    }
                    deadline
                }
            };
            match deadline {
                Some(at) => {
                    tokio::select! { _ = notified => {}, _ = tokio::time::sleep_until(at.into()) => {} }
                }
                None => notified.await,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use canary_agent_runtime::AudioFormat;

    fn control() -> PlaybackControl {
        let control = PlaybackControl::default();
        control.configure(48000);
        control.connect("connection");
        control
    }

    fn chunk(value: i16, frames: usize) -> AudioChunk {
        AudioChunk {
            format: AudioFormat {
                encoding: AudioEncoding::PcmS16Le,
                sample_rate: 48000,
                channels: 1,
            },
            data: (0..frames).flat_map(|_| value.to_le_bytes()).collect(),
        }
    }

    fn average(samples: &[f32]) -> f32 {
        samples.iter().sum::<f32>() / samples.len() as f32
    }

    #[test]
    fn new_responses_queue_without_replacing_audio_or_inserting_padding_gaps() {
        for rate in [44100, 48000] {
            let control = control();
            control.configure(rate);
            control.enqueue("a", chunk(8192, 4800), None).unwrap();
            control.finish("a").unwrap();
            control.enqueue("b", chunk(-8192, 4800), None).unwrap();
            control.finish("b").unwrap();
            let frames = rate as usize / 10;
            let samples = control.render(frames * 2, Instant::now() + Duration::from_secs(1));
            assert!(average(&samples[..frames]) > 0.24);
            assert!(average(&samples[frames..]) < -0.24);
            assert!(
                samples[frames - 8..frames + 8]
                    .iter()
                    .all(|sample| sample.abs() > 0.1),
                "no inserted boundary silence"
            );
            let states = control.states().unwrap();
            assert_eq!(states[0].received_audio_ms, 100);
            assert_eq!(states[1].received_audio_ms, 100);
            assert_eq!(
                states[0].estimated_played_audio_ms, 0,
                "scheduled is not played"
            );
        }
    }

    #[test]
    fn targeted_stop_preserves_other_responses_and_rejects_late_audio() {
        let control = control();
        control.enqueue("a", chunk(8192, 4800), None).unwrap();
        control.enqueue("b", chunk(-8192, 4800), None).unwrap();
        control.finish("b").unwrap();
        assert_eq!(control.stop("connection:a").unwrap().status, "stopped");
        assert_eq!(
            control.stop("connection:a").unwrap().status,
            "already_stopped"
        );
        control.enqueue("a", chunk(8192, 4800), None).unwrap();
        control.finish("a").unwrap();
        let samples = control.render(4800, Instant::now());
        assert!(average(&samples) < -0.24);
        assert_eq!(
            control
                .states()
                .unwrap()
                .iter()
                .find(|s| s.response_id == "a")
                .unwrap()
                .received_audio_ms,
            100
        );
    }

    #[test]
    fn interleaved_chunks_keep_fifo_response_ownership() {
        let control = control();
        control.enqueue("a", chunk(8192, 2400), None).unwrap();
        control.enqueue("b", chunk(-8192, 4800), None).unwrap();
        control.finish("b").unwrap();
        control.enqueue("a", chunk(8192, 2400), None).unwrap();
        control.finish("a").unwrap();
        let samples = control.render(9600, Instant::now());
        assert!(average(&samples[..4800]) > 0.24);
        assert!(average(&samples[4800..]) < -0.24);
    }

    #[test]
    fn stale_stop_after_reconnect_cannot_stop_new_playback_with_same_response_id() {
        let control = control();
        control.enqueue("same", chunk(8192, 4800), None).unwrap();
        control.connect("new");
        control.enqueue("same", chunk(-8192, 4800), None).unwrap();
        control.finish("same").unwrap();
        assert_eq!(
            control.stop("connection:same").unwrap().status,
            "already_stopped"
        );
        assert!(average(&control.render(4800, Instant::now())) < -0.24);
        assert_eq!(control.stop("unknown").unwrap().status, "not_found");
    }

    #[test]
    fn terminal_history_is_bounded_but_pruned_ids_cannot_restart() {
        let control = control();
        for index in 0..48 {
            let id = format!("response-{index:03}");
            control.enqueue(&id, chunk(8192, 480), None).unwrap();
            control.finish(&id).unwrap();
            control.render(480, Instant::now() - Duration::from_secs(1));
            control.feedback();
        }
        assert!(control.states().unwrap().len() <= 32);
        control
            .enqueue("response-000", chunk(8192, 480), None)
            .unwrap();
        assert!(
            control
                .render(480, Instant::now())
                .iter()
                .all(|sample| *sample == 0.0)
        );
    }

    #[test]
    fn stopping_partially_played_response_keeps_next_response_intact() {
        let control = control();
        control.enqueue("a", chunk(8192, 4800), None).unwrap();
        control.finish("a").unwrap();
        control.enqueue("b", chunk(-8192, 4800), None).unwrap();
        control.finish("b").unwrap();
        control.render(2400, Instant::now() - Duration::from_secs(1));
        let stopped = control.stop("connection:a").unwrap();
        assert_eq!(stopped.playback.unwrap().estimated_played_audio_ms, 50);
        assert!(average(&control.render(4800, Instant::now())) < -0.24);
    }

    #[test]
    fn cursor_counts_device_playback_not_queueing_and_retains_scheduled_tail_on_stop() {
        let control = control();
        control.enqueue("a", chunk(8192, 4800), None).unwrap();
        control.finish("a").unwrap();
        let start = Instant::now() + Duration::from_secs(1);
        control.render(4800, start);
        let buffer = control.buffer.lock().unwrap();
        let response = buffer.responses.get("connection:a").unwrap();
        let state = response.state(start + Duration::from_millis(25));
        assert_eq!(state.estimated_played_audio_ms, 25);
        assert_eq!(state.queued_audio_ms, 0);
        assert_eq!(state.scheduled_audio_ms, 75);
        drop(buffer);
        control.stop("connection:a").unwrap();
        let buffer = control.buffer.lock().unwrap();
        let state = buffer
            .responses
            .get("connection:a")
            .unwrap()
            .state(start + Duration::from_millis(100));
        assert_eq!(state.status, "stopped");
        assert_eq!(state.estimated_played_audio_ms, 100);
        assert_eq!(state.scheduled_audio_ms, 0);
    }

    fn target(run: &str) -> Target {
        let (writer, _) = crate::bridge::spawn(
            "python3",
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("python/mock_worker.py"),
        )
        .unwrap();
        Target {
            connection_id: "connection".into(),
            run_id: run.into(),
            turn_id: "turn".into(),
            writer: Arc::new(tokio::sync::Mutex::new(writer)),
        }
    }

    #[tokio::test]
    async fn overlapping_native_responses_are_acknowledged_only_when_each_is_played() {
        let control = control();
        let a = target("a");
        let b = target("b");
        control
            .enqueue(
                "a",
                chunk(8192, 4800),
                Some(Tracking {
                    chunk_id: "a-chunk".into(),
                    target: a.clone(),
                }),
            )
            .unwrap();
        control.audio_done(a).unwrap();
        control
            .enqueue(
                "b",
                chunk(-8192, 4800),
                Some(Tracking {
                    chunk_id: "b-chunk".into(),
                    target: b.clone(),
                }),
            )
            .unwrap();
        control.audio_done(b).unwrap();
        assert!(control.feedback().is_empty());
        control.render(4800, Instant::now() - Duration::from_secs(1));
        assert_eq!(control.feedback().len(), 2);
        assert_eq!(
            control
                .states()
                .unwrap()
                .iter()
                .find(|s| s.response_id == "b")
                .unwrap()
                .estimated_played_audio_ms,
            0
        );
        control.render(4800, Instant::now() - Duration::from_secs(1));
        assert_eq!(control.feedback().len(), 2);
    }

    #[tokio::test]
    async fn provider_stop_before_first_chunk_rejects_late_audio() {
        let control = control();
        let target = target("a");
        control.stop_target("a", &target);
        control
            .enqueue(
                "a",
                chunk(8192, 4800),
                Some(Tracking {
                    chunk_id: "late".into(),
                    target,
                }),
            )
            .unwrap();
        assert!(
            control
                .render(4800, Instant::now())
                .iter()
                .all(|s| *s == 0.0)
        );
        assert_eq!(control.feedback().len(), 1);
    }
}
