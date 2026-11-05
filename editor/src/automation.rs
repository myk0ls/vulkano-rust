//! Replays a text file of UI actions, so the editor can be exercised end to end without
//! a person at the keyboard. Enabled with `--automation <file>`. One step per line:
//!
//! ```text
//! wait 30                   # let frames pass
//! move 400 300              # move the pointer
//! click 400 300             # left click (also: rclick)
//! drag 400 300 500 300      # press, move, release
//! mdrag 400 300 500 300     # the same with the middle button
//! rdown 400 300 / rup       # hold or release the right button
//! key W                     # press and release (ctrl+S, shift+F5, ...)
//! keydown W / keyup W       # hold or release a key
//! type hello                # text input
//! shot /tmp/out.png         # save a screenshot of the next frame
//! resize 1200 700           # resize the window
//! camera x y z dist yaw pitch # place the viewport camera (Y-up pivot, radians)
//! select Name               # select an entity by Name, or fail
//! expect_selected Name      # fail unless this entity is selected (or: none)
//! expect_entities 3         # fail unless the scene has this many entities
//! expect_clean              # fail if there are unsaved changes (also: expect_dirty)
//! expect_component N Comp   # fail unless entity N has this component
//! expect_no_component N Comp
//! mark N / expect_moved N   # fail unless N's position changed since the mark
//! mark_camera / expect_camera_moved
//! expect_log TEXT           # fail unless the Output dock contains TEXT
//! expect_no_log TEXT
//! expect_file PATH TEXT     # fail unless the file contains TEXT
//! write PATH TEXT           # write a file, as an external editor would
//! set_script_editor CMD     # choose the command scripts open with
//! quit
//! ```

use egui::{Event, Key, Modifiers, PointerButton, Pos2, pos2};
use shipyard::{EntityId, World};

use crate::editor::Editor;
use crate::painter::Painter;
use crate::panels::viewport::entity_pose;
use crate::platform::Platform;

pub struct Automation {
    steps: Vec<String>,
    next: usize,
    wait: u32,
    pending_events: Vec<Vec<Event>>,
    marks: std::collections::HashMap<String, nalgebra_glm::Vec3>,
    camera_mark: Option<(nalgebra_glm::Vec3, f32, f32, f32)>,
    /// A window size for the main loop to apply.
    pub resize: Option<(u32, u32)>,
    pub failures: Vec<String>,
}

fn find(editor: &Editor, world: &World, name: &str) -> Option<EntityId> {
    editor.doc.order.iter().copied().find(|e| editor.doc.display_name(world, *e) == name)
}

impl Automation {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        Ok(Self {
            steps: text
                .lines()
                .map(|l| l.split('#').next().unwrap().trim().to_string())
                .filter(|l| !l.is_empty())
                .collect(),
            next: 0,
            wait: 0,
            pending_events: Vec::new(),
            marks: Default::default(),
            camera_mark: None,
            resize: None,
            failures: Vec::new(),
        })
    }

    /// Run due steps for this frame. Returns true when finished.
    pub fn tick(&mut self, editor: &mut Editor, world: &World, platform: &mut Platform, painter: &mut Painter) -> bool {
        // Spread pointer sequences across frames, the way real input arrives.
        if !self.pending_events.is_empty() {
            for event in self.pending_events.remove(0) {
                platform.inject(event);
            }
            return false;
        }
        if self.wait > 0 {
            self.wait -= 1;
            return false;
        }
        // Long imports must not race the steps.
        if editor.busy.is_some() || editor.pending_open.is_some() {
            return false;
        }
        let Some(line) = self.steps.get(self.next).cloned() else {
            return true;
        };
        self.next += 1;
        println!("[auto] {line}");
        let (command, rest) = line.split_once(' ').unwrap_or((line.as_str(), ""));
        let numbers: Vec<f32> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();

        match command {
            "wait" => self.wait = numbers.first().copied().unwrap_or(1.0) as u32,
            "move" => platform.inject(Event::PointerMoved(pos2(numbers[0], numbers[1]))),
            "click" | "rclick" => {
                let at = pos2(numbers[0], numbers[1]);
                let button = if command == "click" { PointerButton::Primary } else { PointerButton::Secondary };
                self.pending_events.push(vec![Event::PointerMoved(at)]);
                self.pending_events.push(vec![press(at, button, true)]);
                self.pending_events.push(vec![press(at, button, false)]);
            }
            "drag" | "mdrag" => {
                let button = if command == "drag" { PointerButton::Primary } else { PointerButton::Middle };
                let (from, to) = (pos2(numbers[0], numbers[1]), pos2(numbers[2], numbers[3]));
                self.pending_events.push(vec![Event::PointerMoved(from)]);
                self.pending_events.push(vec![Event::PointerMoved(from)]);
                self.pending_events.push(vec![press(from, button, true)]);
                for i in 1..=10 {
                    self.pending_events.push(vec![Event::PointerMoved(from + (to - from) * (i as f32 / 10.0))]);
                }
                self.pending_events.push(vec![press(to, button, false)]);
            }
            "rdown" => {
                let at = pos2(numbers[0], numbers[1]);
                self.pending_events.push(vec![Event::PointerMoved(at)]);
                self.pending_events.push(vec![press(at, PointerButton::Secondary, true)]);
            }
            "rup" => self.pending_events.push(vec![press(pos2(0.0, 0.0), PointerButton::Secondary, false)]),
            "key" => match parse_key(rest) {
                Some((key, modifiers)) => {
                    self.pending_events.push(vec![key_event(key, modifiers, true)]);
                    self.pending_events.push(vec![key_event(key, modifiers, false)]);
                }
                None => self.fail(format!("unknown key: {rest}")),
            },
            "keydown" | "keyup" => match parse_key(rest) {
                Some((key, modifiers)) => self.pending_events.push(vec![key_event(key, modifiers, command == "keydown")]),
                None => self.fail(format!("unknown key: {rest}")),
            },
            "type" => platform.inject(Event::Text(rest.to_string())),
            "shot" => painter.request_screenshot(rest.to_string()),
            "camera" => {
                let camera = &mut editor.viewport.camera;
                camera.pivot = nalgebra_glm::vec3(numbers[0], numbers[1], numbers[2]);
                camera.distance = numbers[3];
                camera.yaw = numbers[4];
                camera.pitch = numbers[5];
            }
            "resize" => self.resize = Some((numbers[0] as u32, numbers[1] as u32)),
            "select" => match find(editor, world, rest) {
                Some(entity) => editor.doc.selection = Some(entity),
                None => self.fail(format!("no entity named {rest}")),
            },
            "expect_selected" => {
                let selected = editor
                    .doc
                    .selection
                    .map_or("none".to_string(), |e| editor.doc.display_name(world, e));
                if selected != rest {
                    self.fail(format!("expected {rest} selected, found {selected}"));
                }
            }
            "expect_entities" => {
                let count = editor.doc.order.len();
                if count != numbers[0] as usize {
                    self.fail(format!("expected {} entities, found {count}", numbers[0]));
                }
            }
            "expect_clean" if editor.doc.is_dirty() => self.fail("expected no unsaved changes".into()),
            "expect_dirty" if !editor.doc.is_dirty() => self.fail("expected unsaved changes".into()),
            "expect_clean" | "expect_dirty" => {}
            "expect_component" | "expect_no_component" => {
                let (name, component) = rest.split_once(' ').unwrap_or((rest, ""));
                let has = find(editor, world, name)
                    .is_some_and(|e| editor.registry.def(component).is_some_and(|d| d.has(world, e)));
                if has != (command == "expect_component") {
                    self.fail(format!("{command}: {rest}"));
                }
            }
            "mark" => match find(editor, world, rest).and_then(|e| entity_pose(world, e)) {
                Some(pose) => {
                    self.marks.insert(rest.to_string(), pose.position);
                }
                None => self.fail(format!("nothing to mark: {rest}")),
            },
            "expect_moved" => {
                let position = find(editor, world, rest).and_then(|e| entity_pose(world, e)).map(|p| p.position);
                if position.is_none() || self.marks.get(rest).copied() == position {
                    self.fail(format!("{rest} did not move"));
                }
            }
            "mark_camera" => self.camera_mark = Some(camera_state(editor)),
            "expect_camera_moved" => {
                if self.camera_mark == Some(camera_state(editor)) {
                    self.fail("the camera did not move".into());
                }
            }
            "expect_log" | "expect_no_log" => {
                let found = editor.log.entries.iter().any(|e| e.text.contains(rest));
                if found != (command == "expect_log") {
                    self.fail(format!("{command}: {rest}"));
                }
            }
            "expect_file" => {
                let (path, text) = rest.split_once(' ').unwrap_or((rest, ""));
                let contents = std::fs::read_to_string(path).unwrap_or_default();
                if !contents.contains(text) {
                    self.fail(format!("{path} does not contain: {text}"));
                }
            }
            "write" => {
                let (path, text) = rest.split_once(' ').unwrap_or((rest, ""));
                if let Err(e) = std::fs::write(path, text) {
                    self.fail(format!("could not write {path}: {e}"));
                }
            }
            "set_script_editor" => editor.scripts.editor_command = rest.to_string(),
            "quit" => {
                self.next = self.steps.len();
                return true;
            }
            other => self.fail(format!("unknown step: {other}")),
        }
        false
    }

    fn fail(&mut self, message: String) {
        println!("[auto] FAILED: {message}");
        self.failures.push(message);
    }
}

fn camera_state(editor: &Editor) -> (nalgebra_glm::Vec3, f32, f32, f32) {
    let camera = &editor.viewport.camera;
    (camera.pivot, camera.distance, camera.yaw, camera.pitch)
}

fn press(pos: Pos2, button: PointerButton, pressed: bool) -> Event {
    Event::PointerButton { pos, button, pressed, modifiers: Modifiers::NONE }
}

fn key_event(key: Key, modifiers: Modifiers, pressed: bool) -> Event {
    Event::Key { key, physical_key: None, pressed, repeat: false, modifiers }
}

fn parse_key(text: &str) -> Option<(Key, Modifiers)> {
    let mut modifiers = Modifiers::NONE;
    let mut key = None;
    for part in text.split('+') {
        match part.trim() {
            "ctrl" => modifiers = modifiers | Modifiers::COMMAND,
            "shift" => modifiers = modifiers | Modifiers::SHIFT,
            "alt" => modifiers = modifiers | Modifiers::ALT,
            name => key = Key::from_name(name),
        }
    }
    Some((key?, modifiers))
}
