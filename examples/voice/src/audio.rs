use crate::feedback::{Delivery, Target};
use canary_agent_runtime::{AgentError, AudioChunk, AudioFormat, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{FftFixedInOut, Resampler};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Notify, mpsc, watch};
mod playback;
pub use playback::{PlaybackControl, Tracking};
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
    #[cfg(test)]
    fn reset(&mut self) {
        self.pending.clear();
        self.resampler.reset();
    }

    #[cfg(test)]
    fn output_position(&self, input_frames: u64, input_rate: u32, output_rate: u32) -> u64 {
        input_frames * u64::from(output_rate) / u64::from(input_rate)
            + self.resampler.output_delay() as u64
    }
}

pub struct AudioDevices {
    _input: cpal::Stream,
    _output: cpal::Stream,
    _processing: ProcessingWorker,
    pub microphone: Option<mpsc::Receiver<AudioChunk>>,
    pub errors: watch::Receiver<Option<String>>,
    playback: PlaybackControl,
    pub feedback_changed: Arc<Notify>,
}

impl AudioDevices {
    pub fn open(format: AudioFormat, playback: PlaybackControl) -> Result<Self> {
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
        let feedback_changed = playback.changed.clone();
        playback.configure(output_config.sample_rate().0);
        let output_stream = match output_config.sample_format() {
            cpal::SampleFormat::F32 => build_output::<f32>(
                &output,
                &output_config.into(),
                playback.clone(),
                processing.input.clone(),
            )?,
            cpal::SampleFormat::I16 => build_output::<i16>(
                &output,
                &output_config.into(),
                playback.clone(),
                processing.input.clone(),
            )?,
            cpal::SampleFormat::U16 => build_output::<u16>(
                &output,
                &output_config.into(),
                playback.clone(),
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
            feedback_changed,
        })
    }

    pub fn connect(&self, connection: &str) {
        self.playback.connect(connection);
    }

    pub fn text_delta(&self, response_id: &str, text: &str) {
        self.playback.text_delta(response_id, text);
    }

    pub fn enqueue(
        &self,
        response_id: &str,
        chunk: AudioChunk,
        tracking: Option<Tracking>,
    ) -> Result<()> {
        self.playback.enqueue(response_id, chunk, tracking)
    }

    pub fn interrupt(&self) {
        self.playback.stop_all();
    }

    pub fn finish_response(&self, response_id: &str) -> Result<()> {
        self.playback.finish(response_id)
    }

    pub fn audio_done(&self, target: Target) -> Result<()> {
        self.playback.audio_done(target)
    }

    pub fn stop_target(&self, response_id: &str, target: &Target) {
        self.playback.stop_target(response_id, target);
    }

    pub fn feedback(&self) -> Vec<Delivery> {
        self.playback.feedback()
    }

    pub fn feedback_deadline(&self) -> Option<Instant> {
        self.playback.deadline()
    }

    pub async fn wait_drained(&self) {
        self.playback.wait_drained().await;
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
    playback: PlaybackControl,
    processing: ProcessingInput,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let channels = usize::from(config.channels);
    let device_errors = processing.clone();
    device
        .build_output_stream(
            config,
            move |data: &mut [T], info| {
                let timestamp = info.timestamp();
                let delay = timestamp
                    .playback
                    .duration_since(&timestamp.callback)
                    .unwrap_or_default();
                let rendered = playback.render(data.len() / channels, Instant::now() + delay);
                for (frame, value) in data.chunks_mut(channels).zip(&rendered) {
                    for sample in frame {
                        *sample = T::from_sample_(*value);
                    }
                }
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
