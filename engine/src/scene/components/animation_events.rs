use shipyard::Component;

/// Script functions an animation calls when it reaches a moment, like Godot's call
/// tracks. The function runs on the scripts of the animated entity.
#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct AnimationEvents {
    pub events: Vec<AnimationEvent>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnimationEvent {
    pub clip: String,
    pub time: f32,
    pub function: String,
}

impl AnimationEvents {
    /// The text form used in scene files: `Walk@0.25=footstep; Walk@0.75=footstep`.
    pub fn to_text(&self) -> String {
        self.events
            .iter()
            .map(|e| format!("{}@{:.3}={}", e.clip, e.time, e.function))
            .collect::<Vec<_>>()
            .join("; ")
    }

    pub fn from_text(text: &str) -> Self {
        let events = text
            .split(';')
            .filter_map(|item| {
                let (clip, rest) = item.trim().split_once('@')?;
                let (time, function) = rest.split_once('=')?;
                Some(AnimationEvent {
                    clip: clip.trim().to_string(),
                    time: time.trim().parse().ok()?,
                    function: function.trim().to_string(),
                })
            })
            .filter(|e| !e.function.is_empty())
            .collect();
        Self { events }
    }

    /// Functions to call for the clip's time moving from `from` to `to`, wrapping around
    /// the end when it loops.
    pub fn crossed(&self, clip: &str, from: f32, to: f32) -> Vec<String> {
        self.events
            .iter()
            .filter(|e| e.clip == clip)
            .filter(|e| if to >= from { e.time > from && e.time <= to } else { e.time > from || e.time <= to })
            .map(|e| e.function.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_survive_text() {
        let events = AnimationEvents::from_text("Walk@0.25=footstep; Run@1=shout");
        assert_eq!(events.events.len(), 2);
        assert_eq!(AnimationEvents::from_text(&events.to_text()), events);
    }

    #[test]
    fn crossing_handles_loops() {
        let events = AnimationEvents::from_text("Walk@0.25=a; Walk@0.9=b; Run@0.3=c");
        assert_eq!(events.crossed("Walk", 0.2, 0.3), ["a"]);
        assert_eq!(events.crossed("Walk", 0.8, 0.1), ["b"]);
        assert!(events.crossed("Walk", 0.3, 0.8).is_empty());
    }
}
