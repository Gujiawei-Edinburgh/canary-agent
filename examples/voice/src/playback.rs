//! Application policy: the newest audible response owns the speaker.
use std::collections::HashSet;

#[derive(Default)]
pub struct PlaybackResponses {
    current: Option<String>,
    closed: HashSet<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum AudioDecision {
    Replace,
    Append,
    Ignore,
}

impl PlaybackResponses {
    pub fn audio(&mut self, id: &str) -> AudioDecision {
        if self.closed.contains(id) {
            return AudioDecision::Ignore;
        }
        if self.current.as_deref() == Some(id) {
            return AudioDecision::Append;
        }
        if let Some(previous) = self.current.replace(id.into()) {
            self.closed.insert(previous);
        }
        AudioDecision::Replace
    }

    /// Only the current response may flush or clear the playback buffer.
    pub fn finish(&mut self, id: &str) -> bool {
        self.closed.insert(id.into()) && self.current.as_deref() == Some(id)
    }

    pub fn interrupt(&mut self) {
        if let Some(id) = self.current.take() {
            self.closed.insert(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_audio_replaces_and_late_events_cannot_touch_the_new_buffer() {
        let mut playback = PlaybackResponses::default();
        assert_eq!(playback.audio("old"), AudioDecision::Replace);
        assert_eq!(playback.audio("old"), AudioDecision::Append);
        assert_eq!(playback.audio("new"), AudioDecision::Replace);
        assert_eq!(playback.audio("old"), AudioDecision::Ignore);
        assert!(!playback.finish("old"));
        assert_eq!(playback.audio("new"), AudioDecision::Append);
        assert!(playback.finish("new"));
        assert!(!playback.finish("new"));
        assert_eq!(playback.audio("new"), AudioDecision::Ignore);
        assert_eq!(playback.audio("next"), AudioDecision::Replace);
    }

    #[test]
    fn text_only_completion_does_not_affect_speech() {
        let mut playback = PlaybackResponses::default();
        playback.audio("speech");
        assert!(!playback.finish("text-only"));
        assert_eq!(playback.audio("speech"), AudioDecision::Append);
    }

    #[test]
    fn interruption_rejects_late_audio_but_allows_a_new_response() {
        let mut playback = PlaybackResponses::default();
        playback.audio("speech");
        playback.interrupt();
        assert_eq!(playback.audio("speech"), AudioDecision::Ignore);
        assert!(!playback.finish("speech"));
        assert_eq!(playback.audio("next"), AudioDecision::Replace);
    }
}
