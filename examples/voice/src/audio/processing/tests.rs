use super::*;

struct Signal(u32);
impl Signal {
    fn frame(&mut self) -> Vec<f32> {
        (0..FRAME_SAMPLES)
            .map(|_| {
                self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
                ((self.0 >> 8) as f32 / 16777216.0 - 0.5) * 0.4
            })
            .collect()
    }
}

#[test]
fn removes_delayed_speaker_echo_without_muting_microphone() {
    let mut processor = EchoProcessor::new(48000, 48000, 48000).unwrap();
    let mut signal = Signal(1234);
    let mut room = VecDeque::from(vec![0.0; FRAME_SAMPLES * 3]);
    let mut original_energy = 0.0;
    let mut residual_energy = 0.0;
    for iteration in 0..600 {
        let rendered = signal.frame();
        room.extend(rendered.iter().map(|sample| sample * 0.6));
        let capture: Vec<_> = room.drain(..FRAME_SAMPLES).collect();
        processor.render(rendered).unwrap();
        let cleaned = processor.capture(capture.clone()).unwrap();
        if iteration >= 500 {
            original_energy += capture.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>();
            residual_energy += cleaned.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>();
        }
    }
    let ratio = residual_energy / original_energy;
    assert!(ratio < 0.05, "echo residual energy ratio: {ratio}");
}

#[test]
fn preserves_user_speech_during_assistant_playback() {
    let mut processor = EchoProcessor::new(48000, 48000, 48000).unwrap();
    let mut signal = Signal(1234);
    let mut room = VecDeque::from(vec![0.0; FRAME_SAMPLES * 3]);
    let mut user_energy = 0.0;
    let mut output_energy = 0.0;
    for iteration in 0..600 {
        let rendered = signal.frame();
        room.extend(rendered.iter().map(|sample| sample * 0.6));
        let mut capture: Vec<_> = room.drain(..FRAME_SAMPLES).collect();
        if iteration >= 500 {
            for (index, sample) in capture.iter_mut().enumerate() {
                let user = (2.0
                    * std::f32::consts::PI
                    * 730.0
                    * (iteration * FRAME_SAMPLES + index) as f32
                    / 48000.0)
                    .sin()
                    * 0.15;
                *sample += user;
                user_energy += f64::from(user).powi(2);
            }
        }
        processor.render(rendered).unwrap();
        let cleaned = processor.capture(capture).unwrap();
        if iteration >= 500 {
            output_energy += cleaned.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>();
        }
    }
    let ratio = output_energy / user_energy;
    assert!(
        ratio > 0.25 && ratio < 2.0,
        "near-end speech energy ratio: {ratio}"
    );
}

#[test]
fn processes_fragmented_device_buffers_into_provider_rate_frames() {
    let mut processor = EchoProcessor::new(48000, 48000, 16000).unwrap();
    processor.render(vec![0.0; 200]).unwrap();
    processor.render(vec![0.0; 280]).unwrap();
    assert!(processor.capture(vec![0.0; 479]).unwrap().is_empty());
    assert_eq!(processor.capture(vec![0.0; 1]).unwrap().len(), 160);
}

#[test]
fn idle_worker_shuts_down_via_notification() {
    let worker = ProcessingWorker::start(
        48000,
        48000,
        AudioFormat {
            encoding: canary_agent_runtime::AudioEncoding::PcmS16Le,
            sample_rate: 16000,
            channels: 1,
        },
    )
    .unwrap();
    drop(worker);
}
