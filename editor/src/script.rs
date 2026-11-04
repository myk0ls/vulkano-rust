//! Replays a text file of UI actions, so the editor can be exercised end to end
//! without a person at the keyboard. Enabled with `--script <file>`. One step per line:
//!
//! ```text
//! wait 30                   # let frames pass
//! move 400 300              # move the pointer
//! click 400 300             # left click (also: rclick)
//! drag 400 300 500 300      # press, move, release
//! key W                     # press and release (ctrl+S, shift+F5, ...)
//! type hello                # text input
//! shot /tmp/out.png         # save a screenshot of the next frame
//! select Name               # select a node by name, or fail
//! expect_selected Name      # fail unless this node is selected (or: none)
//! expect_nodes 3            # fail unless the scene has this many nodes
//! expect_clean              # fail if there are unsaved changes (also: expect_dirty)
//! expect_position N x y z   # fail unless node N's local position is close
//! expect_moved N            # fail unless node N has moved since the last `mark N`
//! mark N                    # remember node N's position
//! rdown X Y / rup           # hold or release the right mouse button
//! keydown W / keyup W       # hold or release a key
//! mdrag x0 y0 x1 y1         # drag with the middle mouse button
//! mark_camera               # remember the camera
//! expect_camera_moved       # fail unless the camera changed since mark_camera
//! expect_log TEXT           # fail unless the Output dock contains TEXT
//! resize W H                # resize the window
//! quit
//! ```

use egui::{Event, Key, Modifiers, PointerButton, Pos2, pos2};

use crate::editor::Editor;
use crate::painter::Painter;
use crate::platform::Platform;

pub struct Script {
    steps: Vec<String>,
    next: usize,
    wait: u32,
    pending_events: Vec<Vec<Event>>,
    marks: std::collections::HashMap<String, [f32; 3]>,
    camera_mark: Option<(nalgebra_glm::Vec3, f32, f32, f32)>,
    /// A window size for the main loop to apply.
    pub resize: Option<(u32, u32)>,
    pub failures: Vec<String>,
}

impl Script {
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

    /// Run due steps for this frame. Returns true when the script has finished.
    pub fn tick(&mut self, editor: &mut Editor, platform: &mut Platform, painter: &mut Painter) -> bool {
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
        // Long imports must not race the script.
        if editor.busy.is_some() {
            return false;
        }
        let Some(line) = self.steps.get(self.next).cloned() else {
            return true;
        };
        self.next += 1;
        println!("[script] {line}");
        let (command, rest) = line.split_once(' ').unwrap_or((line.as_str(), ""));
        let numbers: Vec<f32> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();

        match command {
            "wait" => self.wait = numbers.first().copied().unwrap_or(1.0) as u32,
            "move" => platform.inject(Event::PointerMoved(pos2(numbers[0], numbers[1]))),
            "click" | "rclick" => {
                let at = pos2(numbers[0], numbers[1]);
                let button = if command == "click" {
                    PointerButton::Primary
                } else {
                    PointerButton::Secondary
                };
                self.pending_events.push(vec![Event::PointerMoved(at)]);
                self.pending_events.push(vec![press(at, button, true)]);
                self.pending_events.push(vec![press(at, button, false)]);
            }
            "drag" => {
                let (from, to) = (pos2(numbers[0], numbers[1]), pos2(numbers[2], numbers[3]));
                self.pending_events.push(vec![Event::PointerMoved(from)]);
                self.pending_events.push(vec![Event::PointerMoved(from)]);
                self.pending_events.push(vec![press(from, PointerButton::Primary, true)]);
                for i in 1..=10 {
                    self.pending_events
                        .push(vec![Event::PointerMoved(from + (to - from) * (i as f32 / 10.0))]);
                }
                self.pending_events.push(vec![press(to, PointerButton::Primary, false)]);
            }
            "key" => match parse_key(rest) {
                Some((key, modifiers)) => {
                    self.pending_events.push(vec![key_event(key, modifiers, true)]);
                    self.pending_events.push(vec![key_event(key, modifiers, false)]);
                }
                None => self.fail(format!("unknown key: {rest}")),
            },
            "type" => platform.inject(Event::Text(rest.to_string())),
            "shot" => painter.request_screenshot(rest.to_string()),
            "select" => match editor.doc.scene.node_named(rest).map(|n| n.id) {
                Some(id) => editor.doc.selection = Some(id),
                None => self.fail(format!("no node named {rest}")),
            },
            "expect_selected" => {
                let selected = editor
                    .doc
                    .selection
                    .and_then(|id| editor.doc.node(id))
                    .map_or("none", |n| n.name.as_str());
                if selected != rest {
                    self.fail(format!("expected {rest} selected, found {selected}"));
                }
            }
            "expect_nodes" => {
                let count = editor.doc.scene.nodes.len();
                if count != numbers[0] as usize {
                    self.fail(format!("expected {} nodes, found {count}", numbers[0]));
                }
            }
            "expect_clean" if editor.doc.is_dirty() => self.fail("expected no unsaved changes".into()),
            "expect_dirty" if !editor.doc.is_dirty() => self.fail("expected unsaved changes".into()),
            "expect_clean" | "expect_dirty" => {}
            "expect_position" => {
                let name = rest.split_whitespace().next().unwrap_or("");
                let expected = &numbers;
                match editor.doc.scene.node_named(name) {
                    Some(node) => {
                        let p = node.transform.position;
                        if (0..3).any(|i| (p[i] - expected[i]).abs() > 0.01) {
                            self.fail(format!("{name} is at {p:?}, expected {expected:?}"));
                        }
                    }
                    None => self.fail(format!("no node named {name}")),
                }
            }
            "mark" => match editor.doc.scene.node_named(rest) {
                Some(node) => {
                    self.marks.insert(rest.to_string(), node.transform.position);
                }
                None => self.fail(format!("no node named {rest}")),
            },
            "expect_moved" => match editor.doc.scene.node_named(rest) {
                Some(node) if self.marks.get(rest) == Some(&node.transform.position) => {
                    self.fail(format!("{rest} did not move"));
                }
                Some(_) => {}
                None => self.fail(format!("no node named {rest}")),
            },
            "rdown" => {
                let at = pos2(numbers[0], numbers[1]);
                self.pending_events.push(vec![Event::PointerMoved(at)]);
                self.pending_events.push(vec![press(at, PointerButton::Secondary, true)]);
            }
            "rup" => {
                let at = pos2(0.0, 0.0);
                self.pending_events.push(vec![press(at, PointerButton::Secondary, false)]);
            }
            "keydown" | "keyup" => match parse_key(rest) {
                Some((key, modifiers)) => {
                    self.pending_events.push(vec![key_event(key, modifiers, command == "keydown")]);
                }
                None => self.fail(format!("unknown key: {rest}")),
            },
            "mdrag" => {
                let (from, to) = (pos2(numbers[0], numbers[1]), pos2(numbers[2], numbers[3]));
                self.pending_events.push(vec![Event::PointerMoved(from)]);
                self.pending_events.push(vec![press(from, PointerButton::Middle, true)]);
                for i in 1..=10 {
                    self.pending_events
                        .push(vec![Event::PointerMoved(from + (to - from) * (i as f32 / 10.0))]);
                }
                self.pending_events.push(vec![press(to, PointerButton::Middle, false)]);
            }
            "mark_camera" => self.camera_mark = Some(camera_state(editor)),
            "expect_camera_moved" => {
                if self.camera_mark == Some(camera_state(editor)) {
                    self.fail("the camera did not move".into());
                }
            }
            "expect_log" => {
                if !editor.log.entries.iter().any(|e| e.text.contains(rest)) {
                    self.fail(format!("the log does not contain: {rest}"));
                }
            }
            "resize" => self.resize = Some((numbers[0] as u32, numbers[1] as u32)),
            "quit" => {
                self.next = self.steps.len();
                return true;
            }
            other => self.fail(format!("unknown step: {other}")),
        }
        false
    }

    fn fail(&mut self, message: String) {
        println!("[script] FAILED: {message}");
        self.failures.push(message);
    }
}

fn camera_state(editor: &Editor) -> (nalgebra_glm::Vec3, f32, f32, f32) {
    let camera = &editor.viewport.camera;
    (camera.pivot, camera.distance, camera.yaw, camera.pitch)
}

fn press(pos: Pos2, button: PointerButton, pressed: bool) -> Event {
    Event::PointerButton {
        pos,
        button,
        pressed,
        modifiers: Modifiers::NONE,
    }
}

fn key_event(key: Key, modifiers: Modifiers, pressed: bool) -> Event {
    Event::Key {
        key,
        physical_key: None,
        pressed,
        repeat: false,
        modifiers,
    }
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
