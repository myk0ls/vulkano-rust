use std::collections::HashMap;
use std::sync::Arc;

use super::sound::Sound;

pub type VoiceId = u64;

/// One sound playing: where it is, which way it goes, and how loud it is in each ear.
#[derive(Clone, Debug)]
pub struct Voice {
    pub sound: Arc<Sound>,
    /// Position in frames of the sound, between samples.
    cursor: f64,
    /// Playback speed, which also shifts pitch. 1 is normal.
    pub speed: f64,
    pub reversed: bool,
    pub looping: bool,
    pub paused: bool,
    /// Reached the end (or the start, reversed) without looping.
    pub finished: bool,
    /// Started by a script without an entity; removed once finished.
    pub one_shot: bool,
    /// The gains to reach by the end of the next block.
    target: [f32; 2],
    /// The gains the last block ended on. Moving gradually avoids clicks when a 3D sound
    /// or its listener moves.
    current: [f32; 2],
    fresh: bool,
}

impl Voice {
    pub fn new(sound: Arc<Sound>) -> Self {
        Self {
            sound,
            cursor: 0.0,
            speed: 1.0,
            reversed: false,
            looping: false,
            paused: false,
            finished: false,
            one_shot: false,
            target: [1.0, 1.0],
            current: [1.0, 1.0],
            fresh: true,
        }
    }

    pub fn set_gain(&mut self, gain: [f32; 2]) {
        self.target = gain;
        if self.fresh {
            self.current = gain;
        }
    }

    pub fn gain(&self) -> [f32; 2] {
        self.target
    }

    /// The position in seconds.
    pub fn time(&self) -> f64 {
        self.cursor / self.sound.rate as f64
    }

    /// Go to `seconds` into the sound.
    pub fn seek(&mut self, seconds: f64) {
        let last = (self.sound.frames() as f64 - 1.0).max(0.0);
        self.cursor = (seconds * self.sound.rate as f64).clamp(0.0, last);
        self.finished = false;
    }

    /// Go to where playing begins: the first frame, or the last when reversed.
    pub fn seek_to_start(&mut self) {
        self.cursor = if self.reversed { (self.sound.frames() as f64 - 1.0).max(0.0) } else { 0.0 };
        self.finished = false;
    }

    /// Add this voice into interleaved stereo `out`.
    fn render(&mut self, out: &mut [f32], output_rate: u32, master: f32) {
        self.fresh = false;
        let frames = self.sound.frames();
        if frames == 0 {
            self.finished = true;
        }
        if self.paused || self.finished {
            self.current = self.target;
            return;
        }
        let samples = &self.sound.samples;
        let length = frames as f64;
        let direction = if self.reversed { -1.0 } else { 1.0 };
        let step = self.speed * self.sound.rate as f64 / output_rate as f64 * direction;
        let count = out.len() / 2;
        let [from_l, from_r] = self.current;
        let [to_l, to_r] = self.target;
        for (i, frame) in out.chunks_exact_mut(2).enumerate() {
            let t = (i + 1) as f32 / count as f32;
            let gain_l = (from_l + (to_l - from_l) * t) * master;
            let gain_r = (from_r + (to_r - from_r) * t) * master;

            let index = (self.cursor.floor() as usize).min(frames - 1);
            let next = if index + 1 < frames { index + 1 } else if self.looping { 0 } else { index };
            let blend = (self.cursor - index as f64) as f32;
            let left = samples[index * 2] + (samples[next * 2] - samples[index * 2]) * blend;
            let right = samples[index * 2 + 1] + (samples[next * 2 + 1] - samples[index * 2 + 1]) * blend;
            frame[0] += left * gain_l;
            frame[1] += right * gain_r;

            self.cursor += step;
            if self.cursor >= length || self.cursor < 0.0 {
                if self.looping {
                    self.cursor = self.cursor.rem_euclid(length);
                } else {
                    self.cursor = self.cursor.clamp(0.0, length - 1.0);
                    self.finished = true;
                    break;
                }
            }
        }
        self.current = self.target;
    }
}

/// Every playing voice, mixed into one stereo stream.
#[derive(Debug)]
pub struct Mixer {
    pub voices: HashMap<VoiceId, Voice>,
    pub output_rate: u32,
    /// Scales everything.
    pub master: f32,
}

impl Mixer {
    pub fn new(output_rate: u32) -> Self {
        Self { voices: HashMap::new(), output_rate, master: 1.0 }
    }

    /// Mix the next block into `out`, interleaved stereo, which must start silent.
    pub fn render(&mut self, out: &mut [f32]) {
        for voice in self.voices.values_mut() {
            voice.render(out, self.output_rate, self.master);
        }
        // Many loud voices at once are clipped here rather than by the device.
        for sample in out.iter_mut() {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ramp 0, 1, 2, ... in both ears, at the output rate, scaled small.
    fn ramp(frames: usize) -> Arc<Sound> {
        let mono: Vec<f32> = (0..frames).map(|i| i as f32 * 0.01).collect();
        Arc::new(Sound::from_interleaved(&mono, 1, 100))
    }

    fn mix(voice: Voice, frames: usize) -> (Vec<f32>, Voice) {
        let mut mixer = Mixer::new(100);
        mixer.voices.insert(1, voice);
        let mut out = vec![0.0; frames * 2];
        mixer.render(&mut out);
        let left = out.chunks(2).map(|f| (f[0] * 100.0).round()).collect();
        (left, mixer.voices.remove(&1).unwrap())
    }

    #[test]
    fn plays_forwards_and_finishes() {
        let (out, voice) = mix(Voice::new(ramp(4)), 6);
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
        assert!(voice.finished);
    }

    #[test]
    fn plays_backwards_from_the_end() {
        let mut voice = Voice::new(ramp(4));
        voice.reversed = true;
        voice.seek_to_start();
        let (out, voice) = mix(voice, 5);
        assert_eq!(out, [3.0, 2.0, 1.0, 0.0, 0.0]);
        assert!(voice.finished);
    }

    #[test]
    fn loops_both_ways() {
        let mut voice = Voice::new(ramp(3));
        voice.looping = true;
        let (out, voice) = mix(voice, 7);
        assert_eq!(out, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0]);
        assert!(!voice.finished);

        let mut voice = Voice::new(ramp(3));
        voice.looping = true;
        voice.reversed = true;
        voice.seek_to_start();
        let (out, _) = mix(voice, 5);
        assert_eq!(out, [2.0, 1.0, 0.0, 2.0, 1.0]);
    }

    #[test]
    fn speed_skips_frames() {
        let mut voice = Voice::new(ramp(8));
        voice.speed = 2.0;
        let (out, _) = mix(voice, 4);
        assert_eq!(out, [0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn paused_voices_are_silent_and_keep_their_place() {
        let mut voice = Voice::new(ramp(8));
        voice.seek(0.03);
        voice.paused = true;
        let (out, voice) = mix(voice, 3);
        assert_eq!(out, [0.0, 0.0, 0.0]);
        assert!((voice.time() - 0.03).abs() < 1e-9);
    }

    #[test]
    fn panning_keeps_the_centre_at_full_volume() {
        let [l, r] = crate::audio::pan_gains(0.0);
        assert!((l - 1.0).abs() < 1e-5 && (r - 1.0).abs() < 1e-5);
        let [l, r] = crate::audio::pan_gains(1.0);
        assert!(l.abs() < 1e-5 && (r - 1.0).abs() < 1e-5);
    }
}
