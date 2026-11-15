//! Keyframed animation of any component field, like Godot's AnimationPlayer.
//!
//! An [`Animation`] (a `.vanim` file) has tracks. A property track changes one field of
//! one component, on the player's own entity or another entity found by Name, blending
//! between keys; a call track calls script functions on that entity at its keys' times.
//! An [`AnimationPlayer`] lists the animations an entity can play and plays one at a time.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use shipyard::Component;

use crate::scene::scene_file::FieldValue;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interpolation {
    /// Straight from one key to the next.
    #[default]
    Linear,
    /// Easing in and out of each key.
    Smooth,
    /// Holding each key's value until the next.
    Step,
}

pub const INTERPOLATIONS: &[&str] = &["Linear", "Smooth", "Step"];

impl Interpolation {
    pub fn name(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Smooth => "Smooth",
            Self::Step => "Step",
        }
    }

    pub fn from_name(name: &str) -> Self {
        match name {
            "Smooth" => Self::Smooth,
            "Step" => Self::Step,
            _ => Self::Linear,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TrackKind {
    /// Sets `component.field` from the keys' values.
    Property { component: String, field: String, interpolation: Interpolation },
    /// Calls the function named by each key's value (a Text) on the target's scripts.
    Call,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Key {
    pub time: f32,
    pub value: FieldValue,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Track {
    /// Another entity's Name, or empty for the player's own entity.
    pub target: String,
    pub kind: TrackKind,
    /// Kept sorted by time.
    pub keys: Vec<Key>,
}

impl Track {
    pub fn label(&self) -> String {
        let target = if self.target.is_empty() { "self" } else { &self.target };
        match &self.kind {
            TrackKind::Property { component, field, .. } => format!("{target} • {component}.{field}"),
            TrackKind::Call => format!("{target} • call"),
        }
    }

    pub fn sort(&mut self) {
        self.keys.sort_by(|a, b| a.time.total_cmp(&b.time));
    }

    /// Add a key, replacing one at (nearly) the same time.
    pub fn insert(&mut self, time: f32, value: FieldValue) {
        self.keys.retain(|k| (k.time - time).abs() > 1e-4);
        self.keys.push(Key { time, value });
        self.sort();
    }

    /// The value at `time`, blended between the keys around it.
    pub fn sample(&self, time: f32) -> Option<FieldValue> {
        let TrackKind::Property { interpolation, .. } = &self.kind else {
            return None;
        };
        let after = self.keys.iter().position(|k| k.time > time);
        let (a, b) = match after {
            None => return self.keys.last().map(|k| k.value.clone()),
            Some(0) => return self.keys.first().map(|k| k.value.clone()),
            Some(i) => (&self.keys[i - 1], &self.keys[i]),
        };
        let span = (b.time - a.time).max(1e-6);
        let mut t = ((time - a.time) / span).clamp(0.0, 1.0);
        match interpolation {
            Interpolation::Step => return Some(a.value.clone()),
            Interpolation::Smooth => t = t * t * (3.0 - 2.0 * t),
            Interpolation::Linear => {}
        }
        Some(blend(&a.value, &b.value, t))
    }

    /// Functions whose keys the time passed going from `from` to `to`, wrapping around the
    /// end when the animation loops.
    pub fn crossed(&self, from: f32, to: f32) -> Vec<String> {
        if self.kind != TrackKind::Call {
            return Vec::new();
        }
        self.keys
            .iter()
            .filter(|k| if to >= from { k.time > from && k.time <= to } else { k.time > from || k.time <= to })
            .filter_map(|k| k.value.as_text().map(str::to_string))
            .filter(|f| !f.is_empty())
            .collect()
    }
}

/// Numbers and vectors blend; anything else switches halfway.
pub fn blend(a: &FieldValue, b: &FieldValue, t: f32) -> FieldValue {
    match (a, b) {
        (FieldValue::Float(x), FieldValue::Float(y)) => FieldValue::Float(x + (y - x) * t),
        (FieldValue::Int(x), FieldValue::Int(y)) => FieldValue::Int((*x as f32 + (*y - *x) as f32 * t).round() as i64),
        (FieldValue::Vec3(x), FieldValue::Vec3(y)) => FieldValue::Vec3([0, 1, 2].map(|i| x[i] + (y[i] - x[i]) * t)),
        _ if t < 0.5 => a.clone(),
        _ => b.clone(),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Animation {
    /// In seconds.
    pub length: f32,
    pub looping: bool,
    pub tracks: Vec<Track>,
}

impl Default for Animation {
    fn default() -> Self {
        Self { length: 1.0, looping: false, tracks: Vec::new() }
    }
}

impl Animation {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("could not read {path}: {e}"))?;
        let mut animation: Self = ron::from_str(&text).map_err(|e| format!("could not parse {path}: {e}"))?;
        animation.tracks.iter_mut().for_each(Track::sort);
        Ok(animation)
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let config = ron::ser::PrettyConfig::new().depth_limit(4);
        let text = ron::ser::to_string_pretty(self, config).map_err(|e| format!("could not write the animation: {e}"))?;
        std::fs::write(path, text).map_err(|e| format!("could not write {path}: {e}"))
    }
}

/// The name an animation is played by: its file name without the extension.
pub fn animation_name(path: &str) -> String {
    std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

struct Cached {
    modified: Option<SystemTime>,
    animation: Option<Arc<Animation>>,
}

static ANIMATIONS: Mutex<Option<HashMap<String, Cached>>> = Mutex::new(None);

/// The animation in a file, read again when the file changes.
pub fn get(path: &str) -> Option<Arc<Animation>> {
    let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    let mut cache = ANIMATIONS.lock().unwrap();
    let cache = cache.get_or_insert_with(HashMap::new);
    if let Some(cached) = cache.get(path) {
        if cached.modified == modified {
            return cached.animation.clone();
        }
    }
    let animation = match Animation::load(path) {
        Ok(a) => Some(Arc::new(a)),
        Err(e) => {
            eprintln!("[animation] {e}");
            None
        }
    };
    cache.insert(path.to_string(), Cached { modified, animation: animation.clone() });
    animation
}

/// Save an animation and use it at once.
pub fn store(path: &str, animation: &Animation) -> Result<(), String> {
    animation.save(path)?;
    let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    let mut cache = ANIMATIONS.lock().unwrap();
    cache
        .get_or_insert_with(HashMap::new)
        .insert(path.to_string(), Cached { modified, animation: Some(Arc::new(animation.clone())) });
    Ok(())
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct AnimationPlayer {
    /// The `.vanim` files this entity can play, by their names.
    pub animations: Vec<String>,
    /// The name of an animation to start with. Empty starts none.
    pub autoplay: String,
    pub speed: f32,

    pub(crate) current: Option<String>,
    pub(crate) time: f32,
    pub(crate) playing: bool,
    pub(crate) started: bool,
}

impl Default for AnimationPlayer {
    fn default() -> Self {
        Self {
            animations: Vec::new(),
            autoplay: String::new(),
            speed: 1.0,
            current: None,
            time: 0.0,
            playing: false,
            started: false,
        }
    }
}

impl AnimationPlayer {
    /// The file of an animation, by name or path.
    pub fn path_of(&self, name: &str) -> Option<&str> {
        self.animations.iter().map(String::as_str).find(|p| *p == name || animation_name(p) == name)
    }

    /// Play an animation from the start. Returns false if this player doesn't have it.
    pub fn play(&mut self, name: &str) -> bool {
        let Some(path) = self.path_of(name).map(str::to_string) else {
            return false;
        };
        self.current = Some(path);
        self.time = 0.0;
        self.playing = true;
        true
    }

    pub fn stop(&mut self) {
        self.playing = false;
        self.time = 0.0;
    }

    pub fn pause(&mut self) {
        self.playing = false;
    }

    pub fn resume(&mut self) {
        if self.current.is_some() {
            self.playing = true;
        }
    }

    pub fn seek(&mut self, seconds: f32) {
        self.time = seconds.max(0.0);
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    /// The name of the animation playing or last played.
    pub fn current_name(&self) -> String {
        self.current.as_deref().map(animation_name).unwrap_or_default()
    }

    pub fn current_path(&self) -> Option<&str> {
        self.current.as_deref()
    }

    /// Move time on. Returns the time before and after (for call keys) and whether the
    /// animation just ended.
    pub fn advance(&mut self, dt: f32, animation: &Animation) -> (f32, f32, bool) {
        let before = self.time;
        if !self.playing {
            return (before, before, false);
        }
        let length = animation.length.max(0.001);
        let mut time = self.time + dt * self.speed;
        let mut ended = false;
        if animation.looping {
            time = time.rem_euclid(length);
        } else if time >= length {
            time = length;
            self.playing = false;
            ended = true;
        } else if time < 0.0 {
            time = 0.0;
            self.playing = false;
            ended = true;
        }
        self.time = time;
        (before, time, ended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(interpolation: Interpolation) -> Track {
        let mut track = Track {
            target: String::new(),
            kind: TrackKind::Property { component: "Transform".into(), field: "position".into(), interpolation },
            keys: Vec::new(),
        };
        track.insert(1.0, FieldValue::Vec3([2.0, 0.0, 0.0]));
        track.insert(0.0, FieldValue::Vec3([0.0, 0.0, 0.0]));
        track
    }

    #[test]
    fn values_blend_between_keys_and_hold_outside_them() {
        let linear = track(Interpolation::Linear);
        assert_eq!(linear.sample(0.25), Some(FieldValue::Vec3([0.5, 0.0, 0.0])));
        assert_eq!(linear.sample(-1.0), Some(FieldValue::Vec3([0.0, 0.0, 0.0])));
        assert_eq!(linear.sample(5.0), Some(FieldValue::Vec3([2.0, 0.0, 0.0])));
        assert_eq!(track(Interpolation::Step).sample(0.9), Some(FieldValue::Vec3([0.0, 0.0, 0.0])));
        let FieldValue::Vec3([x, ..]) = track(Interpolation::Smooth).sample(0.25).unwrap() else { panic!() };
        assert!(x < 0.5 && x > 0.0, "smooth eases in: {x}");
    }

    #[test]
    fn keys_at_the_same_time_replace_each_other() {
        let mut t = track(Interpolation::Linear);
        t.insert(1.0, FieldValue::Vec3([9.0, 0.0, 0.0]));
        assert_eq!(t.keys.len(), 2);
        assert_eq!(t.sample(1.0), Some(FieldValue::Vec3([9.0, 0.0, 0.0])));
    }

    #[test]
    fn call_keys_fire_once_as_time_passes_them_even_when_looping() {
        let mut calls = Track { target: String::new(), kind: TrackKind::Call, keys: Vec::new() };
        calls.insert(0.5, FieldValue::Text("step".into()));
        assert_eq!(calls.crossed(0.4, 0.6), ["step"]);
        assert!(calls.crossed(0.6, 0.9).is_empty());
        assert_eq!(calls.crossed(0.9, 0.55), ["step"]);
    }

    #[test]
    fn players_end_or_loop() {
        let mut player = AnimationPlayer { animations: vec!["anims/door_open.vanim".into()], ..Default::default() };
        assert!(player.play("door_open"));
        assert!(!player.play("missing"));
        let once = Animation { length: 1.0, looping: false, tracks: Vec::new() };
        assert_eq!(player.advance(0.6, &once), (0.0, 0.6, false));
        assert_eq!(player.advance(0.6, &once), (0.6, 1.0, true));
        assert!(!player.is_playing());

        let looping = Animation { looping: true, ..once };
        player.play("door_open");
        player.advance(1.25, &looping);
        assert!((player.time() - 0.25).abs() < 1e-5 && player.is_playing());
    }

    #[test]
    fn animations_survive_a_save_and_load() {
        let animation = Animation { length: 2.0, looping: true, tracks: vec![track(Interpolation::Smooth)] };
        let path = std::env::temp_dir().join(format!("anim-test-{}.vanim", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        animation.save(&path).unwrap();
        assert_eq!(Animation::load(&path).unwrap(), animation);
        std::fs::remove_file(&path).unwrap();
    }
}
