use canary_agent_runtime::{AgentError, AudioChunk, AudioEncoding, AudioFormat, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{FftFixedInOut, Resampler};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
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
}

pub struct AudioDevices {
    _input: cpal::Stream,
    _output: cpal::Stream,
    _processing: ProcessingWorker,
    pub microphone: Option<mpsc::Receiver<AudioChunk>>,
    pub errors: watch::Receiver<Option<String>>,
    playback: Arc<Mutex<VecDeque<f32>>>,
    playback_converter: Converter,
    output_rate: u32,
    drained: Arc<Notify>,
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
        let playback = Arc::new(Mutex::new(VecDeque::new()));
        let drained = Arc::new(Notify::new());
        let output_rate = output_config.sample_rate().0;
        let output_stream = match output_config.sample_format() {
            cpal::SampleFormat::F32 => build_output::<f32>(
                &output,
                &output_config.into(),
                playback.clone(),
                drained.clone(),
                processing.input.clone(),
            )?,
            cpal::SampleFormat::I16 => build_output::<i16>(
                &output,
                &output_config.into(),
                playback.clone(),
                drained.clone(),
                processing.input.clone(),
            )?,
            cpal::SampleFormat::U16 => build_output::<u16>(
                &output,
                &output_config.into(),
                playback.clone(),
                drained.clone(),
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
            output_rate,
            drained,
        })
    }

    pub fn enqueue(&mut self, chunk: AudioChunk) -> Result<()> {
        if chunk.format.encoding != AudioEncoding::PcmS16Le
            || chunk.format.sample_rate != 24000
            || chunk.format.channels != 1
            || !chunk.data.len().is_multiple_of(2)
        {
            return Err(audio_error("expected mono 24 kHz PCM16 playback"));
        }
        let samples = chunk
            .data
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0);
        let output = self.playback_converter.push(samples)?;
        let mut queue = self
            .playback
            .lock()
            .map_err(|_| audio_error("playback queue poisoned"))?;
        if queue.len() + output.len() > self.output_rate as usize * 300 {
            return Err(audio_error(
                "playback exceeded the example's 30-second buffer",
            ));
        }
        queue.extend(output);
        Ok(())
    }

    pub fn interrupt(&mut self) {
        if let Ok(mut queue) = self.playback.lock() {
            queue.clear();
        }
        self.playback_converter.reset();
        self.drained.notify_one();
    }

    pub fn buffered_samples(&self) -> usize {
        self.playback.lock().map(|q| q.len()).unwrap_or(0)
    }

    pub fn finish_response(&mut self) -> Result<()> {
        let padding = self.playback_converter.resampler.input_frames_next() * 2;
        let tail = self.playback_converter.push(vec![0.0; padding])?;
        self.playback
            .lock()
            .map_err(|_| audio_error("playback queue poisoned"))?
            .extend(tail);
        self.playback_converter.reset();
        Ok(())
    }

    pub async fn wait_drained(&self) {
        loop {
            let notified = self.drained.notified();
            if self.buffered_samples() == 0 {
                return;
            }
            notified.await;
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
    playback: Arc<Mutex<VecDeque<f32>>>,
    drained: Arc<Notify>,
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
            move |data: &mut [T], _| {
                let mut queue = playback.lock().ok();
                let had_audio = queue.as_ref().is_some_and(|q| !q.is_empty());
                let mut rendered = Vec::with_capacity(data.len() / channels);
                for frame in data.chunks_mut(channels) {
                    let value = queue
                        .as_mut()
                        .and_then(|q| q.pop_front())
                        .unwrap_or(0.0)
                        .clamp(-1.0, 1.0);
                    rendered.push(value);
                    for sample in frame {
                        *sample = T::from_sample_(value);
                    }
                }
                if had_audio && queue.as_ref().is_some_and(|q| q.is_empty()) {
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
    fn resamples_native_audio_and_discards_old_samples_on_interruption() {
        let mut converter = Converter::new(48000, 16000).unwrap();
        let samples = converter.push(vec![0.25; 4800]).unwrap();
        assert!(samples.len() >= 1400 && samples.len() <= 1600);
        converter.reset();
        let silence = converter.push(vec![0.0; 4800]).unwrap();
        assert!(silence.iter().all(|sample| sample.abs() < 0.0001));
    }
}
