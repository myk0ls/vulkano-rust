//! Colours and spacing modelled on Godot 4's default editor theme.

use egui::{Color32, CornerRadius, Stroke, Style, Visuals};

pub const BASE: Color32 = Color32::from_rgb(0x36, 0x3d, 0x4a);
pub const PANEL: Color32 = Color32::from_rgb(0x2b, 0x31, 0x3c);
pub const DARK: Color32 = Color32::from_rgb(0x21, 0x26, 0x2e);
pub const INPUT: Color32 = Color32::from_rgb(0x1c, 0x20, 0x27);
pub const HOVER: Color32 = Color32::from_rgb(0x43, 0x4b, 0x5a);
pub const ACTIVE: Color32 = Color32::from_rgb(0x4d, 0x56, 0x67);
pub const ACCENT: Color32 = Color32::from_rgb(0x56, 0x9e, 0xff);
pub const SELECTED: Color32 = Color32::from_rgb(0x3b, 0x5a, 0x85);
pub const TEXT: Color32 = Color32::from_rgb(0xe0, 0xe0, 0xe0);
pub const TEXT_DIM: Color32 = Color32::from_rgb(0x9a, 0xa0, 0xaa);
pub const AXIS_X: Color32 = Color32::from_rgb(0xf5, 0x51, 0x5f);
pub const AXIS_Y: Color32 = Color32::from_rgb(0x8d, 0xd9, 0x3e);
pub const AXIS_Z: Color32 = Color32::from_rgb(0x4b, 0x9b, 0xf5);
pub const NODE_3D: Color32 = Color32::from_rgb(0xfc, 0x7f, 0x7f);
pub const WARNING: Color32 = Color32::from_rgb(0xff, 0xd6, 0x5c);
pub const ERROR: Color32 = Color32::from_rgb(0xff, 0x78, 0x6b);
pub const SELECTION_OUTLINE: Color32 = Color32::from_rgb(0xff, 0xa5, 0x3c);
pub const PLAY: Color32 = Color32::from_rgb(0x8d, 0xd9, 0x7a);

pub fn apply(ctx: &egui::Context) {
    let radius = CornerRadius::same(3);
    let mut visuals = Visuals::dark();
    visuals.override_text_color = Some(TEXT);
    visuals.panel_fill = PANEL;
    visuals.window_fill = PANEL;
    visuals.window_stroke = Stroke::new(1.0, ACTIVE);
    visuals.window_corner_radius = CornerRadius::same(4);
    visuals.menu_corner_radius = radius;
    visuals.extreme_bg_color = INPUT;
    visuals.faint_bg_color = BASE;
    visuals.selection.bg_fill = SELECTED;
    visuals.selection.stroke = Stroke::new(1.0, ACCENT);
    visuals.hyperlink_color = ACCENT;
    visuals.warn_fg_color = WARNING;
    visuals.error_fg_color = ERROR;

    let widgets = &mut visuals.widgets;
    widgets.noninteractive.bg_fill = PANEL;
    widgets.noninteractive.weak_bg_fill = PANEL;
    widgets.noninteractive.bg_stroke = Stroke::new(1.0, DARK);
    widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
    widgets.noninteractive.corner_radius = radius;
    widgets.inactive.bg_fill = BASE;
    widgets.inactive.weak_bg_fill = BASE;
    widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    widgets.inactive.corner_radius = radius;
    widgets.hovered.bg_fill = HOVER;
    widgets.hovered.weak_bg_fill = HOVER;
    widgets.hovered.bg_stroke = Stroke::NONE;
    widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    widgets.hovered.corner_radius = radius;
    widgets.active.bg_fill = ACTIVE;
    widgets.active.weak_bg_fill = ACTIVE;
    widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    widgets.active.corner_radius = radius;
    widgets.open.bg_fill = HOVER;
    widgets.open.weak_bg_fill = HOVER;
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
    style.separator.color_idle = DARK;
    style.separator.color_hovered = ACCENT;
    style.separator.color_dragged = ACCENT;
    style.tab_bar.bg_fill = DARK;
    style.tab_bar.height = 26.0;
    style.tab_bar.hline_color = DARK;
    style.tab_bar.corner_radius = CornerRadius::ZERO;
    style.tab.tab_body.bg_fill = PANEL;
    style.tab.tab_body.stroke = Stroke::NONE;
    style.tab.tab_body.corner_radius = CornerRadius::ZERO;
    let round = CornerRadius { nw: 3, ne: 3, sw: 0, se: 0 };
    style.tab.active.bg_fill = PANEL;
    style.tab.active.text_color = Color32::WHITE;
    style.tab.active.outline_color = PANEL;
    style.tab.active.corner_radius = round;
    style.tab.focused = style.tab.active.clone();
    style.tab.inactive.bg_fill = DARK;
    style.tab.inactive.text_color = TEXT_DIM;
    style.tab.inactive.outline_color = DARK;
    style.tab.inactive.corner_radius = round;
    style.tab.hovered.bg_fill = BASE;
    style.tab.hovered.text_color = TEXT;
    style.tab.hovered.corner_radius = round;
    style
}
