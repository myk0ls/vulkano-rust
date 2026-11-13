//! Components that play a sound. `AudioPlayback` plays it flat, as music or UI sounds;
//! `AudioPlayback3D` plays it from its entity's Transform, quieter with distance and
//! panned toward the side it's on.
//!
//! Control them from Rust with the methods below, or from scripts with `play_audio()`,
//! `pause_audio()`, ...; the engine's audio system (`crate::audio`) does the rest.

use std::ops::{Deref, DerefMut};

use shipyard::Component;

/// Whether a source is making sound.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlayState {
    #[default]
    Stopped,
    Playing,
    Paused,
}

/// Where to move a playing sound, done on the next audio update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Seek {
    /// The beginning, or the end when reversed.
    Start,
    /// Seconds into the sound.
    Time(f32),
}

/// What both audio components share: the sound, its settings, and its playback state.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioSource {
    /// Path to a .wav, .ogg, .mp3 or .flac file.
    pub sound: String,
    pub volume: f32,
    /// Playback speed; 2 is twice as fast and an octave higher.
    pub pitch: f32,
    pub looping: bool,
    /// Start playing when the entity appears.
    pub autoplay: bool,
    /// Play backwards.
    pub reversed: bool,

    pub(crate) state: PlayState,
    pub(crate) seek: Option<Seek>,
    /// Seconds into the sound, as of the last audio update.
    pub(crate) time: f32,
    /// The sound's length in seconds, once loaded.
    pub(crate) length: f32,
    /// Autoplay has been considered.
    pub(crate) started: bool,
    pub(crate) voice: Option<crate::audio::VoiceId>,
    pub(crate) loaded_sound: String,
}

impl Default for AudioSource {
    fn default() -> Self {
        Self {
            sound: String::new(),
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            autoplay: false,
            reversed: false,
            state: PlayState::Stopped,
            seek: None,
            time: 0.0,
            length: 0.0,
            started: false,
            voice: None,
            loaded_sound: String::new(),
        }
    }
}

impl AudioSource {
    pub fn new(sound: &str) -> Self {
        Self { sound: sound.to_string(), ..Default::default() }
    }

    /// Start from the beginning if stopped, carry on if paused, nothing if playing.
    pub fn play(&mut self) {
        match self.state {
            PlayState::Playing => {}
            PlayState::Paused => self.state = PlayState::Playing,
            PlayState::Stopped => {
                self.state = PlayState::Playing;
                self.seek = Some(Seek::Start);
            }
        }
    }

    /// Switch to another sound and play it from the beginning.
    pub fn play_sound(&mut self, sound: &str) {
        self.sound = sound.to_string();
        self.restart();
    }

    /// Stop and go back to the beginning.
    pub fn stop(&mut self) {
        self.state = PlayState::Stopped;
        self.seek = None;
        self.time = 0.0;
    }

    /// Hold the current position.
    pub fn pause(&mut self) {
        if self.state == PlayState::Playing {
            self.state = PlayState::Paused;
        }
    }

    /// Carry on after `pause`.
    pub fn resume(&mut self) {
        if self.state == PlayState::Paused {
            self.state = PlayState::Playing;
        }
    }

    /// Play from the beginning (the end, when reversed), whatever it was doing.
    pub fn restart(&mut self) {
        self.state = PlayState::Playing;
        self.seek = Some(Seek::Start);
        self.time = if self.reversed { self.length } else { 0.0 };
    }

    /// Turn around: a playing sound carries on backwards from where it is.
    pub fn reverse(&mut self) {
        self.reversed = !self.reversed;
    }

    /// Move to `seconds` into the sound. Has no effect while stopped, since `play` starts
    /// from the beginning.
    pub fn seek(&mut self, seconds: f32) {
        if self.state != PlayState::Stopped {
            let seconds = seconds.max(0.0);
            self.seek = Some(Seek::Time(seconds));
            self.time = seconds;
        }
    }

    pub fn state(&self) -> PlayState {
        self.state
    }

    pub fn is_playing(&self) -> bool {
        self.state == PlayState::Playing
    }

    pub fn is_paused(&self) -> bool {
        self.state == PlayState::Paused
    }

    /// Seconds into the sound.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// The sound's length in seconds; 0 until it has loaded, which happens on the first
    /// audio update after `sound` is set.
    pub fn length(&self) -> f32 {
        self.length
    }
}

/// A sound heard the same wherever the listener is.
#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct AudioPlayback {
    pub source: AudioSource,
    /// -1 is fully left, 1 fully right.
    pub pan: f32,
}

/// How a 3D sound gets quieter with distance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Rolloff {
    /// Halves with each doubling of distance past `min_distance`, as sound does; stays at
    /// its `max_distance` level beyond that.
    #[default]
    Inverse,
    /// Full at `min_distance`, silent at `max_distance`.
    Linear,
    /// Drops quickly near the source and eases out to silence at `max_distance`.
    Exponential,
}

pub const ROLLOFFS: &[&str] = &["Inverse", "Linear", "Exponential"];

impl Rolloff {
    pub fn name(self) -> &'static str {
        match self {
            Self::Inverse => "Inverse",
            Self::Linear => "Linear",
            Self::Exponential => "Exponential",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Inverse" => Some(Self::Inverse),
            "Linear" => Some(Self::Linear),
            "Exponential" => Some(Self::Exponential),
            _ => None,
        }
    }

    /// Volume from 0 to 1 at `distance`.
    pub fn attenuation(self, distance: f32, min_distance: f32, max_distance: f32) -> f32 {
        let min = min_distance.max(1e-4);
        let max = max_distance.max(min);
        let d = distance.clamp(min, max);
        match self {
            Self::Inverse => min / d,
            _ if max <= min => 1.0,
            Self::Linear => 1.0 - (d - min) / (max - min),
            Self::Exponential => (1.0 - (d - min) / (max - min)).powi(2),
        }
    }
}

/// A sound that comes from its entity's position.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct AudioPlayback3D {
    pub source: AudioSource,
    /// Full volume within this distance.
    pub min_distance: f32,
    /// See `Rolloff` for what happens past this.
    pub max_distance: f32,
    pub rolloff: Rolloff,
}

impl Default for AudioPlayback3D {
    fn default() -> Self {
        Self { source: AudioSource::default(), min_distance: 1.0, max_distance: 50.0, rolloff: Rolloff::Inverse }
    }
}

impl Deref for AudioPlayback {
    type Target = AudioSource;
    fn deref(&self) -> &AudioSource {
        &self.source
    }
}

impl DerefMut for AudioPlayback {
    fn deref_mut(&mut self) -> &mut AudioSource {
        &mut self.source
    }
}

impl Deref for AudioPlayback3D {
    type Target = AudioSource;
    fn deref(&self) -> &AudioSource {
        &self.source
    }
}

impl DerefMut for AudioPlayback3D {
    fn deref_mut(&mut self) -> &mut AudioSource {
        &mut self.source
    }
}
