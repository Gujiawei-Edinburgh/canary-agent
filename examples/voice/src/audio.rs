use crate::feedback::{Delivery, Target, Tracker};
use canary_agent_runtime::{AgentError, AudioChunk, AudioEncoding, AudioFormat, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{FftFixedInOut, Resampler};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::{Notify, mpsc, watch};
mod processing;
use processing::{ProcessingInput, ProcessingWorker};

struct Converter {
    resampler: FftFixedInOut<f32>,
    pending: VecDeque<f32>,
}

impl Converter {
    fn new(input: u32, output: u32) -> Result<Self> {
        Ok(Self {
            resampler: FftFixedInOut::new(
                input as usize,
                output as usize,
                (input / 100) as usize,
                1,
            )
            .map_err(audio_error)?,
            pending: VecDeque::new(),
        })
    }
    fn push(&mut self, input: impl IntoIterator<Item = f32>) -> Result<Vec<f32>> {
        self.pending.extend(input);
        let mut result = Vec::new();
        while self.pending.len() >= self.resampler.input_frames_next() {
            let block: Vec<_> = self
                .pending
                .drain(..self.resampler.input_frames_next())
                .collect();
            let converted = self
                .resampler
                .process(&[block], None)
                .map_err(audio_error)?;
            result.extend_from_slice(&converted[0]);
        }
        Ok(result)
    }
    fn reset(&mut self) {
        self.pending.clear();
        self.resampler.reset();
    }

    fn output_position(&self, input_frames: u64, input_rate: u32, output_rate: u32) -> u64 {
        input_frames * u64::from(output_rate) / u64::from(input_rate)
            + self.resampler.output_delay() as u64
    }
}

#[derive(Default)]
struct PlaybackBuffer {
    samples: VecDeque<f32>,
    position: u64,
    tracker: Tracker,
    last_delivery: Option<Instant>,
}

pub struct AudioDevices {
    _input: cpal::Stream,
    _output: cpal::Stream,
    _processing: ProcessingWorker,
    pub microphone: Option<mpsc::Receiver<AudioChunk>>,
    pub errors: watch::Receiver<Option<String>>,
    playback: Arc<Mutex<PlaybackBuffer>>,
    playback_converter: Converter,
    playback_input_rate: u32,
    output_rate: u32,
    drained: Arc<Notify>,
    pub feedback_changed: Arc<Notify>,
    input_frames: u64,
}

impl AudioDevices {
    pub fn open(format: AudioFormat) -> Result<Self> {
        let host = cpal::default_host();
        let input = host
            .default_input_device()
            .ok_or_else(|| audio_error("no microphone found"))?;
        let output = host
            .default_output_device()
            .ok_or_else(|| audio_error("no output device found"))?;
        let input_config = input.default_input_config().map_err(audio_error)?;
        let output_config = output.default_output_config().map_err(audio_error)?;
        println!(
            "microphone: {} ({} Hz)",
            input.name().map_err(audio_error)?,
            input_config.sample_rate().0
        );
        println!(
            "speaker: {} ({} Hz)",
            output.name().map_err(audio_error)?,
            output_config.sample_rate().0
        );
        let mut processing = ProcessingWorker::start(
            input_config.sample_rate().0,
            output_config.sample_rate().0,
            format,
        )?;
        let microphone = processing.microphone.take();
        let errors = processing.errors.clone();
        println!("acoustic echo cancellation: enabled (AEC3)");
        let input_stream = match input_config.sample_format() {
            cpal::SampleFormat::F32 => {
                build_input::<f32>(&input, &input_config.into(), processing.input.clone())?
            }
            cpal::SampleFormat::I16 => {
                build_input::<i16>(&input, &input_config.into(), processing.input.clone())?
            }
            cpal::SampleFormat::U16 => {
                build_input::<u16>(&input, &input_config.into(), processing.input.clone())?
            }
            other => {
                return Err(audio_error(format!(
                    "unsupported microphone format: {other}"
                )));
            }
        };
        let playback = Arc::new(Mutex::new(PlaybackBuffer::default()));
        let drained = Arc::new(Notify::new());
        let feedback_changed = Arc::new(Notify::new());
        let output_rate = output_config.sample_rate().0;
        let output_stream = match output_config.sample_format() {
            cpal::SampleFormat::F32 => build_output::<f32>(
                &output,
                &output_config.into(),
                playback.clone(),
                drained.clone(),
                feedback_changed.clone(),
                processing.input.clone(),
            )?,
            cpal::SampleFormat::I16 => build_output::<i16>(
                &output,
                &output_config.into(),
                playback.clone(),
                drained.clone(),
                feedback_changed.clone(),
                processing.input.clone(),
            )?,
            cpal::SampleFormat::U16 => build_output::<u16>(
                &output,
                &output_config.into(),
                playback.clone(),
                drained.clone(),
                feedback_changed.clone(),
                processing.input.clone(),
            )?,
            other => return Err(audio_error(format!("unsupported speaker format: {other}"))),
        };
        output_stream.play().map_err(audio_error)?;
        input_stream.play().map_err(audio_error)?;
        Ok(Self {
            _input: input_stream,
            _output: output_stream,
            _processing: processing,
            microphone,
            errors,
            playback,
            playback_converter: Converter::new(24000, output_rate)?,
            playback_input_rate: 24000,
            output_rate,
            drained,
            feedback_changed,
            input_frames: 0,
        })
    }

    pub fn enqueue(&mut self, chunk: AudioChunk, tracking: Option<(String, Target)>) -> Result<()> {
        if chunk.format.encoding != AudioEncoding::PcmS16Le
            || chunk.format.channels != 1
            || chunk.format.sample_rate == 0
            || !chunk.data.len().is_multiple_of(2)
        {
            return Err(audio_error("expected mono PCM16 playback"));
        }
        if chunk.format.sample_rate != self.playback_input_rate {
            if self.buffered_samples() != 0 || !self.playback_converter.pending.is_empty() {
                return Err(audio_error(
                    "cannot change playback rate with buffered audio",
                ));
            }
            self.playback_converter = Converter::new(chunk.format.sample_rate, self.output_rate)?;
            self.playback_input_rate = chunk.format.sample_rate;
        }
        let frames = (chunk.data.len() / 2) as u64;
        let samples = chunk
            .data
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0);
        let output = self.playback_converter.push(samples)?;
        let mut queue = self
            .playback
            .lock()
            .map_err(|_| audio_error("playback queue poisoned"))?;
        if queue.samples.len() + output.len() > self.output_rate as usize * 300 {
            return Err(audio_error(
                "playback exceeded the example's 30-second buffer",
            ));
        }
        if let Some((id, target)) = tracking {
            let start = self.playback_converter.output_position(
                self.input_frames,
                self.playback_input_rate,
                self.output_rate,
            );
            let end = self.playback_converter.output_position(
                self.input_frames + frames,
                self.playback_input_rate,
                self.output_rate,
            );
            queue.tracker.enqueue(target, id, start, end);
        }
        self.input_frames += frames;
        queue.samples.extend(output);
        Ok(())
    }

    pub fn interrupt(&mut self) {
        if let Ok(mut queue) = self.playback.lock() {
            queue.tracker.stop(Instant::now());
            queue.samples.clear();
            queue.position = 0;
            queue.last_delivery = None;
        }
        self.input_frames = 0;
        self.playback_converter.reset();
        self.drained.notify_one();
        self.feedback_changed.notify_one();
    }

    pub fn buffered_samples(&self) -> usize {
        self.playback.lock().map(|q| q.samples.len()).unwrap_or(0)
    }

    pub fn finish_response(&mut self) -> Result<()> {
        let padding = self.playback_converter.resampler.input_frames_next() * 2;
        let tail = self.playback_converter.push(vec![0.0; padding])?;
        self.playback
            .lock()
            .map_err(|_| audio_error("playback queue poisoned"))?
            .samples
            .extend(tail);
        self.playback_converter.reset();
        self.input_frames = 0;
        Ok(())
    }

    pub fn audio_done(&mut self, target: Target) -> Result<()> {
        let flush = self
            .playback
            .lock()
            .map_err(|_| audio_error("playback queue poisoned"))?
            .tracker
            .audio_done(target);
        if flush {
            self.finish_response()?;
        }
        self.feedback_changed.notify_one();
        Ok(())
    }

    pub fn stop_target(&mut self, target: &Target) {
        let matches = self
            .playback
            .lock()
            .is_ok_and(|mut q| q.tracker.stop_target(target, Instant::now()));
        if matches {
            self.interrupt();
        }
        self.feedback_changed.notify_one();
    }

    pub fn feedback(&self) -> Vec<Delivery> {
        self.playback
            .lock()
            .map(|mut q| q.tracker.drain(Instant::now()))
            .unwrap_or_default()
    }

    pub fn feedback_deadline(&self) -> Option<Instant> {
        self.playback.lock().ok()?.tracker.deadline()
    }

    pub async fn wait_drained(&self) {
        loop {
            let notified = self.drained.notified();
            let deadline = if let Ok(mut queue) = self.playback.lock() {
                queue.tracker.poll(Instant::now());
                if queue.samples.is_empty() {
                    match queue.last_delivery {
                        Some(deadline) if deadline > Instant::now() => Some(deadline),
                        _ => return,
                    }
                } else {
                    None
                }
            } else {
                return;
            };
            if let Some(deadline) = deadline {
                tokio::select! {
                    _ = notified => {},
                    _ = tokio::time::sleep_until(deadline.into()) => {},
                }
            } else {
                notified.await;
            }
        }
    }
}

fn build_input<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    processing: ProcessingInput,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = usize::from(config.channels);
    let device_errors = processing.clone();
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                let mono = data
                    .chunks(channels)
                    .map(|frame| {
                        frame
                            .iter()
                            .map(|sample| <f32 as cpal::FromSample<T>>::from_sample_(*sample))
                            .sum::<f32>()
                            / channels as f32
                    })
                    .collect();
                processing.capture(mono);
            },
            move |error| device_errors.fail(error),
            None,
        )
        .map_err(audio_error)
}

fn build_output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    playback: Arc<Mutex<PlaybackBuffer>>,
    drained: Arc<Notify>,
    feedback_changed: Arc<Notify>,
    processing: ProcessingInput,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let channels = usize::from(config.channels);
    let config_rate = config.sample_rate.0;
    let device_errors = processing.clone();
    device
        .build_output_stream(
            config,
            move |data: &mut [T], info| {
                let mut queue = playback.lock().ok();
                let had_audio = queue.as_ref().is_some_and(|q| !q.samples.is_empty());
                let start = queue.as_ref().map_or(0, |q| q.position);
                let mut consumed = 0;
                let mut rendered = Vec::with_capacity(data.len() / channels);
                for frame in data.chunks_mut(channels) {
                    let value = queue
                        .as_mut()
                        .and_then(|q| {
                            q.samples.pop_front().inspect(|_| {
                                consumed += 1;
                            })
                        })
                        .unwrap_or(0.0)
                        .clamp(-1.0, 1.0);
                    rendered.push(value);
                    for sample in frame {
                        *sample = T::from_sample_(value);
                    }
                }
                if let Some(queue) = &mut queue {
                    queue.position += consumed;
                    let timestamp = info.timestamp();
                    let delay = timestamp
                        .playback
                        .duration_since(&timestamp.callback)
                        .unwrap_or_default();
                    let end = queue.position;
                    if consumed > 0 {
                        queue.last_delivery = Some(
                            Instant::now()
                                + delay
                                + std::time::Duration::from_secs_f64(
                                    consumed as f64 / f64::from(config_rate),
                                ),
                        );
                    }
                    queue
                        .tracker
                        .rendered(start, end, Instant::now() + delay, config_rate);
                }
                if consumed > 0 {
                    feedback_changed.notify_one();
                }
                if had_audio && queue.as_ref().is_some_and(|q| q.samples.is_empty()) {
                    drained.notify_one();
                }
                drop(queue);
                // Feed actual output (including silence), never queued network audio.
                processing.render(rendered);
            },
            move |error| {
                device_errors.fail(error);
            },
            None,
        )
        .map_err(audio_error)
}

fn audio_error(error: impl std::fmt::Display) -> AgentError {
    AgentError::Model(format!("audio: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playback_boundaries_include_resampling_latency_and_flush_tail() {
        for output_rate in [44100, 48000] {
            let mut converter = Converter::new(48000, output_rate).unwrap();
            let end = converter.output_position(4801, 48000, output_rate);
            let start = converter.output_position(0, 48000, output_rate);
            assert_eq!(end - start, 4801 * u64::from(output_rate) / 48000);
            let output = converter.push(vec![0.25; 4801]).unwrap();
            assert!(
                (output.len() as u64) < end,
                "last chunk must not be acknowledged before the filter tail"
            );
            let padding = converter.resampler.input_frames_next() * 2;
            let tail = converter.push(vec![0.0; padding]).unwrap();
            assert!((output.len() + tail.len()) as u64 >= end);
        }
    }
    #[test]
    fn resamples_native_audio_and_discards_old_samples_on_interruption() {
        let mut converter = Converter::new(48000, 16000).unwrap();
        let samples = converter.push(vec![0.25; 4800]).unwrap();
        assert!(samples.len() >= 1400 && samples.len() <= 1600);
        converter.reset();
        let silence = converter.push(vec![0.0; 4800]).unwrap();
        assert!(silence.iter().all(|sample| sample.abs() < 0.0001));
    }
}
