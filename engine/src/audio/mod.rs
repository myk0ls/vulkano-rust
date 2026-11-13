//! Sound playback: decoded sounds, a software mixer fed to SDL's audio device, and the
//! system that drives it from `AudioPlayback` and `AudioPlayback3D` components.
//!
//! Sounds are decoded whole into stereo `f32` when first used, so seeking, looping and
//! playing backwards are just moving a cursor. The mixer runs on SDL's audio thread; the
//! game thread updates it once a frame under a short lock.

pub mod mixer;
pub mod sound;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use nalgebra_glm as glm;
use shipyard::{EntityId, Get, IntoIter, View, ViewMut, World};

pub use mixer::{Mixer, Voice, VoiceId};
pub use sound::Sound;

use crate::scene::components::audio_playback::{AudioPlayback, AudioPlayback3D, AudioSource, PlayState, Rolloff, Seek};
use crate::scene::components::camera::Camera;
use crate::scene::components::transform::Transform;

/// The rate the mixer produces. SDL converts it to whatever the device wants.
pub const OUTPUT_RATE: u32 = 48_000;

/// A sound started by a script that no entity owns, such as a pickup's chime.
#[derive(Clone, Debug, PartialEq)]
pub struct OneShot {
    pub path: String,
    pub volume: f32,
    /// Heard from this point, fading over `max_distance`; `None` plays it flat.
    pub at: Option<[f32; 3]>,
    pub max_distance: f32,
}

/// Where sound is heard from: the active camera.
#[derive(Clone, Copy, Debug)]
pub struct Listener {
    pub position: glm::Vec3,
    pub right: glm::Vec3,
}

impl Default for Listener {
    fn default() -> Self {
        Self { position: glm::vec3(0.0, 0.0, 0.0), right: glm::vec3(1.0, 0.0, 0.0) }
    }
}

impl Listener {
    pub fn from_world(world: &World) -> Self {
        world.run(|cameras: View<Camera>, transforms: View<Transform>| {
            (&cameras, &transforms)
                .iter()
                .find(|(c, _)| c.active)
                .map(|(camera, transform)| {
                    let [x, y, z] = transform.get_position_vector();
                    Self { position: glm::vec3(x, y, z), right: camera.get_right_vector() }
                })
                .unwrap_or_default()
        })
    }

    /// Left and right gains for a sound at `at`.
    pub fn spatial_gains(&self, at: glm::Vec3, min_distance: f32, max_distance: f32, rolloff: Rolloff) -> [f32; 2] {
        let offset = at - self.position;
        let distance = offset.norm();
        let attenuation = rolloff.attenuation(distance, min_distance, max_distance);
        // Close sources sit in the middle of the head rather than flipping sides.
        let pan = if distance > 1e-4 { offset.dot(&self.right) / distance } else { 0.0 };
        let [l, r] = pan_gains(pan * (distance / min_distance.max(1e-4)).min(1.0));
        [l * attenuation, r * attenuation]
    }
}

/// Constant-power panning, scaled so the centre plays at full volume in both ears.
pub fn pan_gains(pan: f32) -> [f32; 2] {
    let angle = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    let s = std::f32::consts::SQRT_2;
    [(angle.cos() * s).min(1.0), (angle.sin() * s).min(1.0)]
}

/// Feeds SDL's audio thread from the shared mixer.
struct MixerCallback(Arc<Mutex<Mixer>>);

impl sdl3::audio::AudioCallback<f32> for MixerCallback {
    fn callback(&mut self, out: &mut [f32]) {
        out.fill(0.0);
        let frames = out.len() / 2 * 2;
        lock(&self.0).render(&mut out[..frames]);
    }
}

fn lock(mixer: &Mutex<Mixer>) -> MutexGuard<'_, Mixer> {
    mixer.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub struct AudioEngine {
    mixer: Arc<Mutex<Mixer>>,
    /// `None` when there is no audio device; the mixer is then advanced by frame time,
    /// so sounds still finish and scripts see the same events.
    stream: Option<sdl3::audio::AudioStreamWithCallback<MixerCallback>>,
    /// Decoded sounds by path. A failed load is kept as `None` so it's reported once.
    sounds: HashMap<String, Option<Arc<Sound>>>,
    next_voice: VoiceId,
}

impl AudioEngine {
    /// Open the default playback device. Without one the game runs silently.
    pub fn new(sdl: &sdl3::Sdl) -> Self {
        let mut engine = Self::headless();
        let opened = sdl.audio().map_err(|e| e.to_string()).and_then(|audio| {
            let spec = sdl3::audio::AudioSpec {
                freq: Some(OUTPUT_RATE as i32),
                channels: Some(2),
                format: Some(sdl3::audio::AudioFormat::f32_sys()),
            };
            let stream = audio
                .open_playback_stream(&spec, MixerCallback(engine.mixer.clone()))
                .map_err(|e| e.to_string())?;
            stream.resume().map_err(|e| e.to_string())?;
            Ok(stream)
        });
        match opened {
            Ok(stream) => engine.stream = Some(stream),
            Err(e) => eprintln!("[audio] no audio device, sound is off: {e}"),
        }
        engine
    }

    /// A mixer with no device, for tools and tests.
    pub fn headless() -> Self {
        Self {
            mixer: Arc::new(Mutex::new(Mixer::new(OUTPUT_RATE))),
            stream: None,
            sounds: HashMap::new(),
            next_voice: 1,
        }
    }

    pub fn has_device(&self) -> bool {
        self.stream.is_some()
    }

    pub fn master_volume(&self) -> f32 {
        lock(&self.mixer).master
    }

    pub fn set_master_volume(&mut self, volume: f32) {
        lock(&self.mixer).master = volume.max(0.0);
    }

    /// Use an already decoded sound for `path`, instead of reading the file.
    pub fn insert_sound(&mut self, path: &str, sound: Sound) {
        self.sounds.insert(path.to_string(), Some(Arc::new(sound)));
    }

    /// The decoded sound at `path`, loading it the first time.
    pub fn sound(&mut self, path: &str) -> Option<Arc<Sound>> {
        if path.is_empty() {
            return None;
        }
        self.sounds
            .entry(path.to_string())
            .or_insert_with(|| match Sound::load(path) {
                Ok(sound) => Some(Arc::new(sound)),
                Err(e) => {
                    eprintln!("[audio] {e}");
                    None
                }
            })
            .clone()
    }

    pub fn play_one_shot(&mut self, shot: &OneShot, listener: &Listener) {
        let Some(sound) = self.sound(&shot.path) else {
            return;
        };
        let gain = match shot.at {
            Some([x, y, z]) => listener.spatial_gains(glm::vec3(x, y, z), 1.0, shot.max_distance, Rolloff::Inverse),
            None => [1.0, 1.0],
        };
        let id = self.next_id();
        let mut voice = Voice::new(sound);
        voice.one_shot = true;
        voice.set_gain([gain[0] * shot.volume, gain[1] * shot.volume]);
        lock(&self.mixer).voices.insert(id, voice);
    }

    fn next_id(&mut self) -> VoiceId {
        self.next_voice += 1;
        self.next_voice
    }

    /// Bring the mixer in line with every audio component, and the components in line
    /// with what has played. `dt` advances the mixer when there's no device. Returns the
    /// entities whose sound reached its end.
    pub fn update(&mut self, world: &mut World, dt: f32) -> Vec<EntityId> {
        if self.stream.is_none() {
            let frames = (dt.max(0.0) * OUTPUT_RATE as f32) as usize;
            let mut scratch = vec![0.0; frames * 2];
            lock(&self.mixer).render(&mut scratch);
        }
        let listener = Listener::from_world(world);
        let mut finished = Vec::new();
        let mut touched = HashSet::new();
        world.run(
            |mut flat: ViewMut<AudioPlayback>, mut spatial: ViewMut<AudioPlayback3D>, transforms: View<Transform>| {
                for (entity, playback) in (&mut flat).iter().with_id() {
                    let gain = pan_gains(playback.pan);
                    if self.sync(&mut playback.source, gain, &mut touched) {
                        finished.push(entity);
                    }
                }
                for (entity, playback) in (&mut spatial).iter().with_id() {
                    let at = (&transforms).get(entity).map_or([0.0; 3], |t| t.get_position_vector());
                    let gain = listener.spatial_gains(
                        glm::vec3(at[0], at[1], at[2]),
                        playback.min_distance,
                        playback.max_distance,
                        playback.rolloff,
                    );
                    if self.sync(&mut playback.source, gain, &mut touched) {
                        finished.push(entity);
                    }
                }
            },
        );
        // Voices of removed components and entities, and finished one-shots, go.
        lock(&self.mixer).voices.retain(|id, voice| if voice.one_shot { !voice.finished } else { touched.contains(id) });
        finished.sort_by_key(|id| id.inner());
        finished.dedup();
        finished
    }

    /// Update one source's voice. Returns true when it just finished.
    fn sync(&mut self, source: &mut AudioSource, gain: [f32; 2], touched: &mut HashSet<VoiceId>) -> bool {
        if !source.started {
            source.started = true;
            if source.autoplay {
                source.play();
            }
        }
        // Load as soon as the sound is set, so its length is known and playing doesn't stall.
        if source.loaded_sound != source.sound {
            source.loaded_sound = source.sound.clone();
            source.length = self.sound(&source.sound).map_or(0.0, |s| s.duration());
            // Changing the sound while it plays starts the new one from the beginning.
            if source.voice.take().is_some() && source.state != PlayState::Stopped {
                source.seek.get_or_insert(Seek::Start);
            }
        }

        let mut just_finished = false;
        if let Some(id) = source.voice {
            let mut mixer = lock(&self.mixer);
            match mixer.voices.get(&id) {
                Some(voice) => {
                    source.time = voice.time() as f32;
                    // A restart asked for this frame wins over reaching the end.
                    if voice.finished && source.seek.is_none() {
                        mixer.voices.remove(&id);
                        source.voice = None;
                        source.state = PlayState::Stopped;
                        source.time = 0.0;
                        just_finished = true;
                    }
                }
                None => source.voice = None,
            }
        }

        if source.state == PlayState::Stopped {
            if let Some(id) = source.voice.take() {
                lock(&self.mixer).voices.remove(&id);
            }
            source.seek = None;
            source.time = 0.0;
            return just_finished;
        }

        let id = match source.voice {
            Some(id) => id,
            None => {
                let Some(sound) = self.sound(&source.sound) else {
                    source.state = PlayState::Stopped;
                    source.seek = None;
                    return just_finished;
                };
                let id = self.next_id();
                let mut voice = Voice::new(sound);
                // A new voice starts at the beginning (or the end, reversed) unless seeking.
                source.seek.get_or_insert(Seek::Start);
                voice.reversed = source.reversed;
                voice.set_gain([gain[0] * source.volume, gain[1] * source.volume]);
                lock(&self.mixer).voices.insert(id, voice);
                source.voice = Some(id);
                id
            }
        };
        touched.insert(id);

        let mut mixer = lock(&self.mixer);
        let Some(voice) = mixer.voices.get_mut(&id) else {
            return just_finished;
        };
        voice.reversed = source.reversed;
        match source.seek.take() {
            Some(Seek::Start) => voice.seek_to_start(),
            Some(Seek::Time(t)) => voice.seek(t as f64),
            None => {}
        }
        voice.speed = source.pitch.max(0.01) as f64;
        voice.looping = source.looping;
        voice.paused = source.state == PlayState::Paused;
        voice.set_gain([gain[0] * source.volume, gain[1] * source.volume]);
        source.time = voice.time() as f32;
        just_finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One second of a steady tone, at the output rate.
    fn engine() -> AudioEngine {
        let mut engine = AudioEngine::headless();
        engine.insert_sound("tone.wav", Sound::from_interleaved(&vec![0.5; OUTPUT_RATE as usize], 1, OUTPUT_RATE));
        engine
    }

    fn source(world: &World, entity: EntityId) -> AudioSource {
        world.get::<&AudioPlayback>(entity).unwrap().source.clone()
    }

    fn change(world: &World, entity: EntityId, f: impl FnOnce(&mut AudioSource)) {
        f(&mut world.get::<&mut AudioPlayback>(entity).unwrap().source);
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn autoplay_plays_to_the_end_and_reports_it() {
        let mut audio = engine();
        let mut world = World::new();
        let mut playback = AudioPlayback { source: AudioSource::new("tone.wav"), pan: 0.0 };
        playback.autoplay = true;
        let entity = world.add_entity(playback);

        assert!(audio.update(&mut world, 0.0).is_empty());
        assert!(source(&world, entity).is_playing());
        assert!(close(source(&world, entity).length(), 1.0));
        audio.update(&mut world, 0.5);
        assert!(close(source(&world, entity).time(), 0.5));
        assert_eq!(audio.update(&mut world, 0.6), [entity]);
        assert_eq!(source(&world, entity).state(), PlayState::Stopped);
        assert!(lock(&audio.mixer).voices.is_empty());
    }

    #[test]
    fn pause_holds_and_resume_carries_on() {
        let mut audio = engine();
        let mut world = World::new();
        let entity = world.add_entity(AudioPlayback { source: AudioSource::new("tone.wav"), pan: 0.0 });
        change(&world, entity, AudioSource::play);
        audio.update(&mut world, 0.0);
        audio.update(&mut world, 0.25);
        change(&world, entity, AudioSource::pause);
        audio.update(&mut world, 0.0);
        audio.update(&mut world, 0.5);
        assert!(close(source(&world, entity).time(), 0.25));
        change(&world, entity, AudioSource::resume);
        audio.update(&mut world, 0.0);
        audio.update(&mut world, 0.25);
        assert!(close(source(&world, entity).time(), 0.5));
    }

    #[test]
    fn reverse_turns_around_and_restart_goes_to_the_end() {
        let mut audio = engine();
        let mut world = World::new();
        let entity = world.add_entity(AudioPlayback { source: AudioSource::new("tone.wav"), pan: 0.0 });
        change(&world, entity, AudioSource::play);
        audio.update(&mut world, 0.0);
        audio.update(&mut world, 0.5);
        change(&world, entity, AudioSource::reverse);
        audio.update(&mut world, 0.0);
        audio.update(&mut world, 0.2);
        assert!(close(source(&world, entity).time(), 0.3));

        change(&world, entity, AudioSource::restart);
        audio.update(&mut world, 0.0);
        assert!(close(source(&world, entity).time(), 1.0));
        audio.update(&mut world, 0.4);
        assert!(close(source(&world, entity).time(), 0.6));
        // Played backwards, it ends at the start.
        assert_eq!(audio.update(&mut world, 0.7), [entity]);
    }

    #[test]
    fn stop_and_restart_after_the_end() {
        let mut audio = engine();
        let mut world = World::new();
        let entity = world.add_entity(AudioPlayback { source: AudioSource::new("tone.wav"), pan: 0.0 });
        change(&world, entity, AudioSource::play);
        audio.update(&mut world, 0.0);
        audio.update(&mut world, 0.3);
        change(&world, entity, AudioSource::stop);
        audio.update(&mut world, 0.0);
        assert!(lock(&audio.mixer).voices.is_empty());
        assert_eq!(source(&world, entity).time(), 0.0);
        change(&world, entity, AudioSource::play);
        audio.update(&mut world, 0.0);
        audio.update(&mut world, 0.1);
        assert!(close(source(&world, entity).time(), 0.1));
    }

    #[test]
    fn looping_never_finishes() {
        let mut audio = engine();
        let mut world = World::new();
        let mut source = AudioSource::new("tone.wav");
        source.looping = true;
        source.autoplay = true;
        let entity = world.add_entity(AudioPlayback { source, pan: 0.0 });
        audio.update(&mut world, 0.0);
        assert!(audio.update(&mut world, 0.75).is_empty());
        assert!(audio.update(&mut world, 0.75).is_empty());
        assert!(close(self::source(&world, entity).time(), 0.5));
    }

    #[test]
    fn removed_entities_go_quiet() {
        let mut audio = engine();
        let mut world = World::new();
        let mut source = AudioSource::new("tone.wav");
        source.autoplay = true;
        let entity = world.add_entity(AudioPlayback { source, pan: 0.0 });
        audio.update(&mut world, 0.0);
        assert_eq!(lock(&audio.mixer).voices.len(), 1);
        world.delete_entity(entity);
        audio.update(&mut world, 0.0);
        assert!(lock(&audio.mixer).voices.is_empty());
    }

    #[test]
    fn spatial_sounds_fade_and_pan() {
        let listener = Listener::default();
        let near = listener.spatial_gains(glm::vec3(0.0, 0.0, 1.0), 1.0, 50.0, Rolloff::Inverse);
        let far = listener.spatial_gains(glm::vec3(0.0, 0.0, 10.0), 1.0, 50.0, Rolloff::Inverse);
        assert!(close(near[0], 1.0) && close(near[1], 1.0));
        assert!(close(far[0], 0.1) && close(far[1], 0.1));
        let right = listener.spatial_gains(glm::vec3(5.0, 0.0, 0.0), 1.0, 50.0, Rolloff::Linear);
        assert!(right[1] > 0.5 && right[0] < 0.01);
        let gone = listener.spatial_gains(glm::vec3(0.0, 0.0, 60.0), 1.0, 50.0, Rolloff::Linear);
        assert_eq!(gone, [0.0, 0.0]);
    }
}
