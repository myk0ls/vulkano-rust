//! In-game user interface: UI elements are entities with a [`UiElement`] component, drawn
//! with egui on top of the scene. The editor uses the same code to preview them.

pub mod painter;
pub mod platform;

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use shipyard::{Component, EntityId, IntoIter, View, World};

use crate::scene::scene_file::Hidden;

pub const KINDS: &[&str] = &["Label", "Button", "Panel", "ProgressBar"];
pub const ANCHORS: &[&str] = &[
    "TopLeft", "Top", "TopRight", "Left", "Center", "Right", "BottomLeft", "Bottom", "BottomRight",
];

/// A piece of on-screen UI. Positions are in pixels from an anchor point on the screen,
/// so a button anchored BottomRight stays in that corner at any window size.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct UiElement {
    /// One of [`KINDS`].
    pub kind: String,
    /// One of [`ANCHORS`].
    pub anchor: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub text: String,
    pub font_size: f32,
    pub text_color: [f32; 3],
    pub background: [f32; 3],
    pub background_opacity: f32,
    /// How full a ProgressBar is, from 0 to 1.
    pub value: f32,
    /// Higher layers draw on top.
    pub layer: i64,
}

impl Default for UiElement {
    fn default() -> Self {
        Self {
            kind: "Label".into(),
            anchor: "TopLeft".into(),
            x: 20.0,
            y: 20.0,
            width: 200.0,
            height: 40.0,
            text: "Label".into(),
            font_size: 22.0,
            text_color: [1.0, 1.0, 1.0],
            background: [0.1, 0.1, 0.1],
            background_opacity: 0.0,
            value: 0.5,
            layer: 0,
        }
    }
}

fn color(rgb: [f32; 3], alpha: f32) -> Color32 {
    let [r, g, b] = rgb.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
    Color32::from_rgba_unmultiplied(r, g, b, (alpha.clamp(0.0, 1.0) * 255.0).round() as u8)
}

impl UiElement {
    /// Where this element sits inside `screen`.
    pub fn rect(&self, screen: Rect) -> Rect {
        let index = ANCHORS.iter().position(|a| *a == self.anchor).unwrap_or(0);
        let fraction = Vec2::new((index % 3) as f32 * 0.5, (index / 3) as f32 * 0.5);
        let anchor_point = screen.min + screen.size() * fraction;
        let size = Vec2::new(self.width.max(1.0), self.height.max(1.0));
        // The element's matching point sits on the anchor, and offsets point inward:
        // a right-anchored element moves left as x grows.
        let min = anchor_point + Vec2::new(self.x, self.y) * self.offset_direction() - size * fraction;
        Rect::from_min_size(min, size)
    }

    /// +1 or -1 per axis: which way positive offsets move the element on screen.
    pub fn offset_direction(&self) -> Vec2 {
        let index = ANCHORS.iter().position(|a| *a == self.anchor).unwrap_or(0);
        let sign = |third: usize| if third == 2 { -1.0 } else { 1.0 };
        Vec2::new(sign(index % 3), sign(index / 3))
    }
}

/// What happened while drawing the UI.
#[derive(Default)]
pub struct UiOutput {
    /// Buttons clicked this frame.
    pub pressed: Vec<EntityId>,
    /// Where each element was drawn, topmost last.
    pub rects: Vec<(EntityId, Rect)>,
}

/// Draw every visible UI element into `screen`. With `interactive`, buttons respond to the
/// pointer; the editor's preview passes false.
pub fn draw(ui: &mut egui::Ui, screen: Rect, world: &World, interactive: bool) -> UiOutput {
    let mut elements: Vec<(EntityId, UiElement)> = world.run(|elements: View<UiElement>, hidden: View<Hidden>| {
        (&elements, !&hidden).iter().with_id().map(|(id, (e, _))| (id, e.clone())).collect()
    });
    elements.sort_by_key(|(id, e)| (e.layer, id.inner()));

    let painter = ui.painter_at(screen);
    let mut output = UiOutput::default();
    for (id, element) in elements {
        let rect = element.rect(screen);
        output.rects.push((id, rect));
        let radius = CornerRadius::same(6);
        let text_color = color(element.text_color, 1.0);
        let font = FontId::proportional(element.font_size.max(4.0));
        let background = color(element.background, element.background_opacity);

        match element.kind.as_str() {
            "Button" => {
                let response = ui.interact(rect, ui.id().with(("ui_element", id.inner())), if interactive { Sense::click() } else { Sense::hover() });
                let mut fill = color(element.background, element.background_opacity.max(0.6));
                if interactive && response.is_pointer_button_down_on() {
                    fill = fill.gamma_multiply(0.7);
                } else if interactive && response.hovered() {
                    fill = fill.linear_multiply(1.3);
                }
                painter.rect_filled(rect, radius, fill);
                painter.rect_stroke(rect, radius, Stroke::new(1.0, text_color.gamma_multiply(0.4)), egui::StrokeKind::Inside);
                painter.text(rect.center(), Align2::CENTER_CENTER, &element.text, font, text_color);
                if interactive && response.clicked() {
                    output.pressed.push(id);
                }
            }
            "Panel" => {
                painter.rect_filled(rect, radius, background);
                if !element.text.is_empty() {
                    painter.text(rect.center(), Align2::CENTER_CENTER, &element.text, font, text_color);
                }
            }
            "ProgressBar" => {
                painter.rect_filled(rect, radius, color(element.background, element.background_opacity.max(0.5)));
                let mut fill = rect;
                fill.max.x = rect.min.x + rect.width() * element.value.clamp(0.0, 1.0);
                painter.rect_filled(fill, radius, text_color);
                if !element.text.is_empty() {
                    painter.text(rect.center(), Align2::CENTER_CENTER, &element.text, font, Color32::WHITE);
                }
            }
            _ => {
                if element.background_opacity > 0.0 {
                    painter.rect_filled(rect, radius, background);
                }
                // A little shadow keeps labels readable over any background.
                let at = Pos2::new(rect.min.x + 4.0, rect.center().y);
                painter.text(at + Vec2::new(1.5, 1.5), Align2::LEFT_CENTER, &element.text, font.clone(), Color32::from_black_alpha(160));
                painter.text(at, Align2::LEFT_CENTER, &element.text, font, text_color);
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn element(anchor: &str, x: f32, y: f32) -> UiElement {
        UiElement { anchor: anchor.into(), x, y, width: 100.0, height: 50.0, ..Default::default() }
    }

    #[test]
    fn anchors_keep_elements_inside_their_corner() {
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        assert_eq!(element("TopLeft", 10.0, 20.0).rect(screen).min, Pos2::new(10.0, 20.0));
        assert_eq!(element("BottomRight", 10.0, 20.0).rect(screen).max, Pos2::new(790.0, 580.0));
        assert_eq!(element("Center", 0.0, 0.0).rect(screen).center(), Pos2::new(400.0, 300.0));
        assert_eq!(element("Top", 0.0, 5.0).rect(screen).center().x, 400.0);
        // Offsets still apply to centered elements.
        assert_eq!(element("Center", 30.0, -10.0).rect(screen).center(), Pos2::new(430.0, 290.0));
    }

    #[test]
    fn the_screen_rect_offsets_everything() {
        let screen = Rect::from_min_size(Pos2::new(300.0, 100.0), Vec2::new(800.0, 600.0));
        assert_eq!(element("TopLeft", 10.0, 20.0).rect(screen).min, Pos2::new(310.0, 120.0));
    }

    #[test]
    fn clicking_a_button_reports_it_pressed() {
        let mut world = World::new();
        let button = world.add_entity(UiElement {
            kind: "Button".into(),
            anchor: "Center".into(),
            x: 0.0,
            y: 0.0,
            ..Default::default()
        });
        world.add_entity(UiElement::default()); // A label, which can't be pressed.

        let ctx = egui::Context::default();
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let center = Pos2::new(400.0, 300.0);
        let mut frame = |events: Vec<egui::Event>| {
            let input = egui::RawInput { screen_rect: Some(screen), events, ..Default::default() };
            let mut pressed = Vec::new();
            let _ = ctx.run_ui(input, |root| {
                egui::CentralPanel::default().frame(egui::Frame::NONE).show(root, |panel| {
                    pressed = draw(panel, screen, &world, true).pressed;
                });
            });
            pressed
        };
        let press = |down| egui::Event::PointerButton {
            pos: center,
            button: egui::PointerButton::Primary,
            pressed: down,
            modifiers: Default::default(),
        };
        assert!(frame(vec![egui::Event::PointerMoved(center)]).is_empty());
        assert!(frame(vec![press(true)]).is_empty());
        assert_eq!(frame(vec![press(false)]), vec![button]);
    }

    #[test]
    fn the_editor_preview_never_presses_buttons() {
        let mut world = World::new();
        world.add_entity(UiElement { kind: "Button".into(), anchor: "Center".into(), x: 0.0, y: 0.0, ..Default::default() });
        let ctx = egui::Context::default();
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let center = Pos2::new(400.0, 300.0);
        let mut any = false;
        for events in [
            vec![egui::Event::PointerMoved(center)],
            vec![egui::Event::PointerButton { pos: center, button: egui::PointerButton::Primary, pressed: true, modifiers: Default::default() }],
            vec![egui::Event::PointerButton { pos: center, button: egui::PointerButton::Primary, pressed: false, modifiers: Default::default() }],
        ] {
            let input = egui::RawInput { screen_rect: Some(screen), events, ..Default::default() };
            let _ = ctx.run_ui(input, |root| {
                any |= !draw(root, screen, &world, false).pressed.is_empty();
            });
        }
        assert!(!any);
    }
}
