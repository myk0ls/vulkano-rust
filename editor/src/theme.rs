//! Editor colours and spacing. Gruvbox is the default.

use std::sync::atomic::{AtomicU8, Ordering};

use egui::{Color32, CornerRadius, Stroke, Style, Visuals};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Theme {
    Gruvbox,
    GodotDark,
}

impl Theme {
    pub const ALL: [Theme; 2] = [Theme::Gruvbox, Theme::GodotDark];

    pub fn label(self) -> &'static str {
        match self {
            Theme::Gruvbox => "Gruvbox",
            Theme::GodotDark => "Godot Dark",
        }
    }

    fn palette(self) -> &'static Palette {
        match self {
            Theme::Gruvbox => &GRUVBOX,
            Theme::GodotDark => &GODOT_DARK,
        }
    }
}

struct Palette {
    base: Color32,
    panel: Color32,
    dark: Color32,
    input: Color32,
    hover: Color32,
    active: Color32,
    accent: Color32,
    selected: Color32,
    text: Color32,
    text_dim: Color32,
    axis_x: Color32,
    axis_y: Color32,
    axis_z: Color32,
    entity: Color32,
    warning: Color32,
    error: Color32,
    selection_outline: Color32,
    play: Color32,
    collider: Color32,
}

const fn rgb(hex: u32) -> Color32 {
    Color32::from_rgb((hex >> 16) as u8, (hex >> 8) as u8, hex as u8)
}

/// https://github.com/morhetz/gruvbox, dark variant.
const GRUVBOX: Palette = Palette {
    base: rgb(0x3c3836),      // bg1
    panel: rgb(0x282828),     // bg0
    dark: rgb(0x1d2021),      // bg0_h
    input: rgb(0x1d2021),     // bg0_h
    hover: rgb(0x504945),     // bg2
    active: rgb(0x665c54),    // bg3
    accent: rgb(0xfabd2f),    // yellow
    selected: rgb(0x504945),  // bg2
    text: rgb(0xebdbb2),      // fg1
    text_dim: rgb(0xa89984),  // gray
    axis_x: rgb(0xfb4934),    // red
    axis_y: rgb(0xb8bb26),    // green
    axis_z: rgb(0x83a598),    // blue
    entity: rgb(0xfe8019),    // orange
    warning: rgb(0xfabd2f),   // yellow
    error: rgb(0xfb4934),     // red
    selection_outline: rgb(0xfe8019), // orange
    play: rgb(0xb8bb26),      // green
    collider: rgb(0x8ec07c),  // aqua
};

const GODOT_DARK: Palette = Palette {
    base: rgb(0x363d4a),
    panel: rgb(0x2b313c),
    dark: rgb(0x21262e),
    input: rgb(0x1c2027),
    hover: rgb(0x434b5a),
    active: rgb(0x4d5667),
    accent: rgb(0x569eff),
    selected: rgb(0x3b5a85),
    text: rgb(0xe0e0e0),
    text_dim: rgb(0x9aa0aa),
    axis_x: rgb(0xf5515f),
    axis_y: rgb(0x8dd93e),
    axis_z: rgb(0x4b9bf5),
    entity: rgb(0xfc7f7f),
    warning: rgb(0xffd65c),
    error: rgb(0xff786b),
    selection_outline: rgb(0xffa53c),
    play: rgb(0x8dd97a),
    collider: rgb(0x6ec8f0),
};

static CURRENT: AtomicU8 = AtomicU8::new(0);

pub fn current() -> Theme {
    Theme::ALL[CURRENT.load(Ordering::Relaxed) as usize]
}

macro_rules! colors {
    ($($name:ident),* $(,)?) => {
        $(pub fn $name() -> Color32 { current().palette().$name })*
    };
}

colors!(
    base, panel, dark, input, hover, active, accent, selected, text, text_dim, axis_x, axis_y,
    axis_z, entity, warning, error, selection_outline, play, collider,
);

/// Apply a saved theme by name, falling back to the default.
pub fn apply_named(ctx: &egui::Context, name: &str) {
    let theme = Theme::ALL.into_iter().find(|t| t.label() == name).unwrap_or(Theme::Gruvbox);
    apply(ctx, theme);
}

pub fn apply(ctx: &egui::Context, theme: Theme) {
    let index = Theme::ALL.iter().position(|t| *t == theme).unwrap_or(0);
    CURRENT.store(index as u8, Ordering::Relaxed);

    let radius = CornerRadius::same(3);
    let mut visuals = Visuals::dark();
    visuals.override_text_color = Some(text());
    visuals.panel_fill = panel();
    visuals.window_fill = panel();
    visuals.window_stroke = Stroke::new(1.0, active());
    visuals.window_corner_radius = CornerRadius::same(4);
    visuals.menu_corner_radius = radius;
    visuals.extreme_bg_color = input();
    visuals.faint_bg_color = base();
    visuals.selection.bg_fill = selected();
    visuals.selection.stroke = Stroke::new(1.0, accent());
    visuals.hyperlink_color = accent();
    visuals.warn_fg_color = warning();
    visuals.error_fg_color = error();

    let widgets = &mut visuals.widgets;
    widgets.noninteractive.bg_fill = panel();
    widgets.noninteractive.weak_bg_fill = panel();
    widgets.noninteractive.bg_stroke = Stroke::new(1.0, dark());
    widgets.noninteractive.fg_stroke = Stroke::new(1.0, text());
    widgets.noninteractive.corner_radius = radius;
    widgets.inactive.bg_fill = base();
    widgets.inactive.weak_bg_fill = base();
    widgets.inactive.fg_stroke = Stroke::new(1.0, text());
    widgets.inactive.corner_radius = radius;
    widgets.hovered.bg_fill = hover();
    widgets.hovered.weak_bg_fill = hover();
    widgets.hovered.bg_stroke = Stroke::NONE;
    widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    widgets.hovered.corner_radius = radius;
    widgets.active.bg_fill = active();
    widgets.active.weak_bg_fill = active();
    widgets.active.bg_stroke = Stroke::new(1.0, accent());
    widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    widgets.active.corner_radius = radius;
    widgets.open.bg_fill = hover();
    widgets.open.weak_bg_fill = hover();
    widgets.open.corner_radius = radius;
    ctx.set_visuals(visuals);

    ctx.global_style_mut(|style: &mut Style| {
        style.spacing.item_spacing = egui::vec2(6.0, 4.0);
        style.spacing.button_padding = egui::vec2(6.0, 3.0);
        style.spacing.interact_size.y = 22.0;
        style.spacing.window_margin = egui::Margin::same(8);
        style.interaction.tooltip_delay = 0.4;
    });
}

pub fn dock_style(ctx: &egui::Context) -> egui_dock::Style {
    let mut style = egui_dock::Style::from_egui(&ctx.global_style());
    style.main_surface_border_stroke = Stroke::NONE;
    style.main_surface_border_rounding = CornerRadius::ZERO;
    style.separator.width = 3.0;
    style.separator.color_idle = dark();
    style.separator.color_hovered = accent();
    style.separator.color_dragged = accent();
    style.tab_bar.bg_fill = dark();
    style.tab_bar.height = 26.0;
    style.tab_bar.hline_color = dark();
    style.tab_bar.corner_radius = CornerRadius::ZERO;
    style.tab.tab_body.bg_fill = panel();
    style.tab.tab_body.stroke = Stroke::NONE;
    style.tab.tab_body.corner_radius = CornerRadius::ZERO;
    let round = CornerRadius { nw: 3, ne: 3, sw: 0, se: 0 };
    style.tab.active.bg_fill = panel();
    style.tab.active.text_color = accent();
    style.tab.active.outline_color = panel();
    style.tab.active.corner_radius = round;
    style.tab.focused = style.tab.active.clone();
    style.tab.inactive.bg_fill = dark();
    style.tab.inactive.text_color = text_dim();
    style.tab.inactive.outline_color = dark();
    style.tab.inactive.corner_radius = round;
    style.tab.hovered.bg_fill = base();
    style.tab.hovered.text_color = text();
    style.tab.hovered.corner_radius = round;
    style
}
