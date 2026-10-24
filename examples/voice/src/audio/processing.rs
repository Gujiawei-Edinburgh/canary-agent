use super::{Converter, audio_error};
use canary_agent_runtime::{AudioChunk, AudioFormat, Result};
use sonora::config::{EchoCanceller, HighPassFilter};
use sonora::{AudioProcessing, Config, StreamConfig};
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc as blocking,
};
use std::thread::JoinHandle;
use tokio::sync::{mpsc, watch};

const PROCESSING_RATE: u32 = 48_000;
const FRAME_SAMPLES: usize = 480;

enum Frame {
    Capture(Vec<f32>),
    Render(Vec<f32>),
    Stop,
}

/// Hardware callbacks only submit audio; AEC and resampling run on one worker.
#[derive(Clone)]
pub(super) struct ProcessingInput {
    sender: blocking::SyncSender<Frame>,
    stopped: Arc<AtomicBool>,
    errors: watch::Sender<Option<String>>,
}

impl ProcessingInput {
    pub(super) fn capture(&self, samples: Vec<f32>) {
        self.submit(Frame::Capture(samples));
    }
    pub(super) fn render(&self, samples: Vec<f32>) {
        self.submit(Frame::Render(samples));
    }

    fn submit(&self, frame: Frame) {
        if self.stopped.load(Ordering::Acquire) {
            return;
        }
        if self.sender.try_send(frame).is_err() {
            self.fail("audio processing queue overflowed or closed");
        }
    }

    pub(super) fn fail(&self, error: impl std::fmt::Display) {
        self.stopped.store(true, Ordering::Release);
        let _ = self.errors.send(Some(error.to_string()));
        let _ = self.sender.try_send(Frame::Stop);
    }
}

pub(super) struct ProcessingWorker {
    pub(super) input: ProcessingInput,
    pub(super) microphone: Option<mpsc::Receiver<AudioChunk>>,
    pub(super) errors: watch::Receiver<Option<String>>,
    thread: Option<JoinHandle<()>>,
}

impl ProcessingWorker {
    pub(super) fn start(input_rate: u32, output_rate: u32, format: AudioFormat) -> Result<Self> {
        let mut processor = EchoProcessor::new(input_rate, output_rate, format.sample_rate)?;
        let (sender, events) = blocking::sync_channel(64);
        let (microphone_tx, microphone) = mpsc::channel(32);
        let (errors_tx, errors) = watch::channel(None);
        let stopped = Arc::new(AtomicBool::new(false));
        let worker_stopped = stopped.clone();
        let worker_errors = errors_tx.clone();
        let thread = std::thread::Builder::new()
            .name("voice-aec".into())
            .spawn(move || {
                let result = (|| -> Result<()> {
                    while let Ok(event) = events.recv() {
                        if worker_stopped.load(Ordering::Acquire) {
                            break;
                        }
                        match event {
                            Frame::Render(samples) => processor.render(samples)?,
                            Frame::Capture(samples) => {
                                let samples = processor.capture(samples)?;
                                if samples.is_empty() {
                                    continue;
                                }
                                let data = samples
                                    .into_iter()
                                    .flat_map(|sample| {
                                        ((sample.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes()
                                    })
                                    .collect();
                                microphone_tx
                                    .try_send(AudioChunk {
                                        format: format.clone(),
                                        data,
                                    })
                                    .map_err(|_| {
                                        audio_error(
                                            "processed microphone queue overflowed or closed",
                                        )
                                    })?;
                            }
                            Frame::Stop => break,
                        }
                    }
                    Ok(())
                })();
                worker_stopped.store(true, Ordering::Release);
                if let Err(error) = result {
                    let _ = worker_errors.send(Some(error.to_string()));
                }
            })?;
        Ok(Self {
            input: ProcessingInput {
                sender,
                stopped,
                errors: errors_tx,
            },
            microphone: Some(microphone),
            errors,
            thread: Some(thread),
        })
    }
}

impl Drop for ProcessingWorker {
    fn drop(&mut self) {
        self.input.stopped.store(true, Ordering::Release);
        // Wake an idle worker; a full queue already guarantees a wakeup.
        let _ = self.input.sender.try_send(Frame::Stop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

enum RateConverter {
    Identity,
    Resample(Box<Converter>),
}
impl RateConverter {
    fn new(input: u32, output: u32) -> Result<Self> {
        if input == output {
            Ok(Self::Identity)
        } else {
            Ok(Self::Resample(Box::new(Converter::new(input, output)?)))
        }
    }
    fn push(&mut self, input: Vec<f32>) -> Result<Vec<f32>> {
        match self {
            Self::Identity => Ok(input),
            Self::Resample(converter) => converter.push(input),
        }
    }
}

struct EchoProcessor {
    apm: AudioProcessing,
    capture_rate: RateConverter,
    render_rate: RateConverter,
    model_rate: RateConverter,
    capture_pending: VecDeque<f32>,
    render_pending: VecDeque<f32>,
}

impl EchoProcessor {
    fn new(input_rate: u32, output_rate: u32, model_rate: u32) -> Result<Self> {
        let config = Config {
            echo_canceller: Some(EchoCanceller::default()),
            high_pass_filter: Some(HighPassFilter::default()),
            ..Config::default()
        };
        let apm = AudioProcessing::builder()
            .config(config)
            .capture_config(StreamConfig::new(PROCESSING_RATE, 1))
            .render_config(StreamConfig::new(PROCESSING_RATE, 1))
            .build();
        Ok(Self {
            apm,
            capture_rate: RateConverter::new(input_rate, PROCESSING_RATE)?,
            render_rate: RateConverter::new(output_rate, PROCESSING_RATE)?,
            model_rate: RateConverter::new(PROCESSING_RATE, model_rate)?,
            capture_pending: VecDeque::new(),
            render_pending: VecDeque::new(),
        })
    }

    fn render(&mut self, samples: Vec<f32>) -> Result<()> {
        self.render_pending.extend(self.render_rate.push(samples)?);
        while self.render_pending.len() >= FRAME_SAMPLES {
            let input: Vec<_> = self.render_pending.drain(..FRAME_SAMPLES).collect();
            let mut output = vec![0.0; FRAME_SAMPLES];
            self.apm
                .process_render_f32(&[&input], &mut [&mut output])
                .map_err(audio_error)?;
        }
        Ok(())
    }

    fn capture(&mut self, samples: Vec<f32>) -> Result<Vec<f32>> {
        self.capture_pending
            .extend(self.capture_rate.push(samples)?);
        let mut cleaned = Vec::new();
        while self.capture_pending.len() >= FRAME_SAMPLES {
            let input: Vec<_> = self.capture_pending.drain(..FRAME_SAMPLES).collect();
            let mut output = vec![0.0; FRAME_SAMPLES];
            self.apm
                .process_capture_f32(&[&input], &mut [&mut output])
                .map_err(audio_error)?;
            cleaned.extend(output);
        }
        self.model_rate.push(cleaned)
    }
}

#[cfg(test)]
mod tests;
