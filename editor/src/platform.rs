//! Translates SDL3 events into egui input, and egui's requests back to SDL.

use egui::{CursorIcon, Event as EguiEvent, Key, Modifiers, PointerButton, Pos2, RawInput, Rect, Vec2};
use sdl3::VideoSubsystem;
use sdl3::event::{Event, WindowEvent};
use sdl3::keyboard::{Keycode, Mod};
use sdl3::mouse::{Cursor, MouseButton, SystemCursor};

pub struct Platform {
    video: VideoSubsystem,
    start: std::time::Instant,
    events: Vec<EguiEvent>,
    modifiers: Modifiers,
    pointer: Pos2,
    cursor: Option<Cursor>,
    cursor_icon: CursorIcon,
    /// While the mouse is grabbed for free-look, motion goes here instead of to the UI.
    pub relative_mode: bool,
    pub relative_motion: Vec2,
}

impl Platform {
    pub fn new(video: VideoSubsystem) -> Self {
        Self {
            video,
            start: std::time::Instant::now(),
            events: Vec::new(),
            modifiers: Modifiers::NONE,
            pointer: Pos2::ZERO,
            cursor: None,
            cursor_icon: CursorIcon::Default,
            relative_mode: false,
            relative_motion: Vec2::ZERO,
        }
    }

    /// Queue an event as if it came from the user. Used by editor scripts.
    pub fn inject(&mut self, event: EguiEvent) {
        if let EguiEvent::PointerMoved(pos) = event {
            self.pointer = pos;
        }
        self.events.push(event);
    }

    pub fn handle_event(&mut self, event: &Event) {
        match event {
            Event::MouseMotion { x, y, xrel, yrel, .. } => {
                if self.relative_mode {
                    self.relative_motion += Vec2::new(*xrel, *yrel);
                } else {
                    self.pointer = Pos2::new(*x, *y);
                    self.events.push(EguiEvent::PointerMoved(self.pointer));
                }
            }
            Event::MouseButtonDown { mouse_btn, x, y, .. } => {
                self.pointer_button(*mouse_btn, *x, *y, true)
            }
            Event::MouseButtonUp { mouse_btn, x, y, .. } => {
                self.pointer_button(*mouse_btn, *x, *y, false)
            }
            Event::MouseWheel { x, y, .. } => {
                self.events.push(EguiEvent::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: Vec2::new(*x, *y),
                    phase: egui::TouchPhase::Move,
                    modifiers: self.modifiers,
                });
            }
            Event::KeyDown { keycode, keymod, repeat, .. } => {
                self.set_modifiers(modifiers_from(*keymod));
                self.key(*keycode, true, *repeat);
            }
            Event::KeyUp { keycode, keymod, .. } => {
                self.set_modifiers(modifiers_from(*keymod));
                self.key(*keycode, false, false);
            }
            Event::TextInput { text, .. } => {
                // Control characters and Ctrl shortcuts aren't text.
                if !self.modifiers.ctrl && !text.chars().all(char::is_control) {
                    self.events.push(EguiEvent::Text(text.clone()));
                }
            }
            Event::Window { win_event, .. } => match win_event {
                WindowEvent::FocusGained => {
                    self.events.push(EguiEvent::WindowFocused(true));
                }
                WindowEvent::FocusLost => {
                    self.events.push(EguiEvent::WindowFocused(false));
                }
                WindowEvent::MouseLeave => self.events.push(EguiEvent::PointerGone),
                _ => {}
            },
            _ => {}
        }
    }

    fn set_modifiers(&mut self, modifiers: Modifiers) {
        if modifiers != self.modifiers {
            self.modifiers = modifiers;
            self.events.push(EguiEvent::ModifiersChanged(modifiers));
        }
    }

    fn pointer_button(&mut self, button: MouseButton, x: f32, y: f32, pressed: bool) {
        let button = match button {
            MouseButton::Left => PointerButton::Primary,
            MouseButton::Right => PointerButton::Secondary,
            MouseButton::Middle => PointerButton::Middle,
            MouseButton::X1 => PointerButton::Extra1,
            MouseButton::X2 => PointerButton::Extra2,
            _ => return,
        };
        // In relative mode the reported position is meaningless, so keep the last one.
        if !self.relative_mode {
            self.pointer = Pos2::new(x, y);
        }
        self.events.push(EguiEvent::PointerButton {
            pos: self.pointer,
            button,
            pressed,
            modifiers: self.modifiers,
        });
    }

    fn key(&mut self, keycode: Option<Keycode>, pressed: bool, repeat: bool) {
        let Some(key) = keycode.and_then(key_from) else {
            return;
        };
        if pressed && self.modifiers.ctrl {
            match key {
                Key::C => self.events.push(EguiEvent::Copy),
                Key::X => self.events.push(EguiEvent::Cut),
                Key::V => {
                    if let Ok(text) = self.video.clipboard().clipboard_text() {
                        self.events.push(EguiEvent::Paste(text));
                    }
                }
                _ => {}
            }
        }
        self.events.push(EguiEvent::Key {
            key,
            physical_key: None,
            pressed,
            repeat,
            modifiers: self.modifiers,
        });
    }

    pub fn take_input(&mut self, window_size: (u32, u32)) -> RawInput {
        RawInput {
            screen_rect: Some(Rect::from_min_size(
                Pos2::ZERO,
                Vec2::new(window_size.0 as f32, window_size.1 as f32),
            )),
            time: Some(self.start.elapsed().as_secs_f64()),
            events: std::mem::take(&mut self.events),
            ..Default::default()
        }
    }

    pub fn apply_output(&mut self, output: egui::PlatformOutput) {
        for command in output.commands {
            if let egui::OutputCommand::CopyText(text) = command {
                let _ = self.video.clipboard().set_clipboard_text(&text);
            }
        }
        if output.cursor_icon != self.cursor_icon || self.cursor.is_none() {
            self.cursor_icon = output.cursor_icon;
            if let Ok(cursor) = Cursor::from_system(system_cursor(output.cursor_icon)) {
                cursor.set();
                self.cursor = Some(cursor);
            }
        }
    }
}

fn modifiers_from(keymod: Mod) -> Modifiers {
    let ctrl = keymod.intersects(Mod::LCTRLMOD | Mod::RCTRLMOD);
    Modifiers {
        alt: keymod.intersects(Mod::LALTMOD | Mod::RALTMOD),
        ctrl,
        shift: keymod.intersects(Mod::LSHIFTMOD | Mod::RSHIFTMOD),
        mac_cmd: false,
        command: ctrl,
    }
}

fn system_cursor(icon: CursorIcon) -> SystemCursor {
    match icon {
        CursorIcon::Text | CursorIcon::VerticalText => SystemCursor::IBeam,
        CursorIcon::PointingHand | CursorIcon::Grab | CursorIcon::Grabbing => SystemCursor::Hand,
        CursorIcon::Crosshair => SystemCursor::Crosshair,
        CursorIcon::Wait => SystemCursor::Wait,
        CursorIcon::Progress => SystemCursor::WaitArrow,
        CursorIcon::Move | CursorIcon::AllScroll => SystemCursor::SizeAll,
        CursorIcon::NotAllowed | CursorIcon::NoDrop => SystemCursor::No,
        CursorIcon::ResizeHorizontal | CursorIcon::ResizeColumn => SystemCursor::SizeWE,
        CursorIcon::ResizeVertical | CursorIcon::ResizeRow => SystemCursor::SizeNS,
        CursorIcon::ResizeNeSw => SystemCursor::SizeNESW,
        CursorIcon::ResizeNwSe => SystemCursor::SizeNWSE,
        _ => SystemCursor::Arrow,
    }
}

fn key_from(keycode: Keycode) -> Option<Key> {
    Some(match keycode {
        Keycode::A => Key::A,
        Keycode::B => Key::B,
        Keycode::C => Key::C,
        Keycode::D => Key::D,
        Keycode::E => Key::E,
        Keycode::F => Key::F,
        Keycode::G => Key::G,
        Keycode::H => Key::H,
        Keycode::I => Key::I,
        Keycode::J => Key::J,
        Keycode::K => Key::K,
        Keycode::L => Key::L,
        Keycode::M => Key::M,
        Keycode::N => Key::N,
        Keycode::O => Key::O,
        Keycode::P => Key::P,
        Keycode::Q => Key::Q,
        Keycode::R => Key::R,
        Keycode::S => Key::S,
        Keycode::T => Key::T,
        Keycode::U => Key::U,
        Keycode::V => Key::V,
        Keycode::W => Key::W,
        Keycode::X => Key::X,
        Keycode::Y => Key::Y,
        Keycode::Z => Key::Z,
        Keycode::_0 | Keycode::Kp0 => Key::Num0,
        Keycode::_1 | Keycode::Kp1 => Key::Num1,
        Keycode::_2 | Keycode::Kp2 => Key::Num2,
        Keycode::_3 | Keycode::Kp3 => Key::Num3,
        Keycode::_4 | Keycode::Kp4 => Key::Num4,
        Keycode::_5 | Keycode::Kp5 => Key::Num5,
        Keycode::_6 | Keycode::Kp6 => Key::Num6,
        Keycode::_7 | Keycode::Kp7 => Key::Num7,
        Keycode::_8 | Keycode::Kp8 => Key::Num8,
        Keycode::_9 | Keycode::Kp9 => Key::Num9,
        Keycode::F1 => Key::F1,
        Keycode::F2 => Key::F2,
        Keycode::F3 => Key::F3,
        Keycode::F4 => Key::F4,
        Keycode::F5 => Key::F5,
        Keycode::F6 => Key::F6,
        Keycode::F7 => Key::F7,
        Keycode::F8 => Key::F8,
        Keycode::F9 => Key::F9,
        Keycode::F10 => Key::F10,
        Keycode::F11 => Key::F11,
        Keycode::F12 => Key::F12,
        Keycode::Return | Keycode::KpEnter => Key::Enter,
        Keycode::Escape => Key::Escape,
        Keycode::Backspace => Key::Backspace,
        Keycode::Tab => Key::Tab,
        Keycode::Space => Key::Space,
        Keycode::Delete => Key::Delete,
        Keycode::Insert => Key::Insert,
        Keycode::Home => Key::Home,
        Keycode::End => Key::End,
        Keycode::PageUp => Key::PageUp,
        Keycode::PageDown => Key::PageDown,
        Keycode::Left => Key::ArrowLeft,
        Keycode::Right => Key::ArrowRight,
        Keycode::Up => Key::ArrowUp,
        Keycode::Down => Key::ArrowDown,
        Keycode::Minus | Keycode::KpMinus => Key::Minus,
        Keycode::Equals => Key::Equals,
        Keycode::Period | Keycode::KpPeriod => Key::Period,
        Keycode::Comma => Key::Comma,
        Keycode::Slash => Key::Slash,
        _ => return None,
    })
}
