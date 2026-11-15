//! A window for editing a UI theme (`.vtheme`). Changes are saved as they're made, so UI
//! elements using the theme update in the 3D view while editing.

use egui::{RichText, Ui};
use vulkano_engine::ui::theme::{self, Style, UiTheme};

use crate::editor::Editor;
use crate::panels::filesystem;

pub struct ThemeEditor {
    pub path: String,
    pub theme: UiTheme,
}

impl ThemeEditor {
    pub fn open(path: &str) -> Result<Self, String> {
        Ok(Self { path: path.to_string(), theme: UiTheme::load(path)? })
    }
}

/// Make a new theme file with the default look.
pub fn create(path: &std::path::Path) -> Result<(), String> {
    UiTheme::default().save(&path.to_string_lossy())
}

pub fn window(ctx: &egui::Context, editor: &mut Editor) {
    let Some(mut state) = editor.theme_editor.take() else {
        return;
    };
    let mut open = true;
    let mut changed = false;
    let fonts = editor.files.files_with_extensions(&["ttf", "otf"]);
    egui::Window::new(format!("UI Theme: {}", filesystem::file_name(&state.path)))
        .id(egui::Id::new("theme_editor"))
        .open(&mut open)
        .default_width(360.0)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().max_height(560.0).show(ui, |ui| {
                changed |= theme_ui(ui, &mut state.theme, &fonts);
            });
            ui.separator();
            ui.label(
                RichText::new("Give a UiElement this theme in its Theme field. Custom Colors keeps the element's own colours.")
                    .small()
                    .color(crate::theme::text_dim()),
            );
        });
    if changed {
        if let Err(e) = theme::store(&state.path, &state.theme) {
            editor.log.error(e);
        }
    }
    if open {
        editor.theme_editor = Some(state);
    }
}

fn theme_ui(ui: &mut Ui, theme: &mut UiTheme, fonts: &[String]) -> bool {
    let mut changed = false;
    egui::Grid::new("theme_grid").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
        ui.label("Font");
        let shown = if theme.font.is_empty() { "[built-in]".to_string() } else { filesystem::file_name(&theme.font) };
        egui::ComboBox::from_id_salt("theme_font").selected_text(shown).width(200.0).show_ui(ui, |ui| {
            if ui.selectable_label(theme.font.is_empty(), "[built-in]").clicked() {
                theme.font.clear();
                changed = true;
            }
            for font in fonts {
                if ui.selectable_label(&theme.font == font, font.as_str()).clicked() {
                    theme.font = font.clone();
                    changed = true;
                }
            }
        });
        ui.end_row();
        if fonts.is_empty() {
            ui.label("");
            ui.label(RichText::new("Put .ttf or .otf files in the project to use them.").small().color(crate::theme::text_dim()));
            ui.end_row();
        }
        changed |= number(ui, "Font scale", &mut theme.font_scale, 0.1..=5.0, 0.01);
        changed |= number(ui, "Corner radius", &mut theme.corner_radius, 0.0..=64.0, 0.1);
        changed |= number(ui, "Border width", &mut theme.border_width, 0.0..=16.0, 0.05);
        changed |= color(ui, "Border colour", &mut theme.border_color);
        changed |= number(ui, "Border opacity", &mut theme.border_opacity, 0.0..=1.0, 0.01);
        ui.label("Text shadow");
        changed |= ui.checkbox(&mut theme.text_shadow, "").changed();
        ui.end_row();
    });
    for (title, style) in [
        ("Labels", &mut theme.label),
        ("Buttons", &mut theme.button),
        ("Panels", &mut theme.panel),
        ("Progress bars", &mut theme.progress),
    ] {
        ui.add_space(4.0);
        ui.label(RichText::new(title).strong());
        egui::Grid::new(("theme_style", title)).num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            changed |= style_ui(ui, style);
        });
    }
    ui.add_space(4.0);
    ui.label(RichText::new("Button states and bar fill").strong());
    egui::Grid::new("theme_states").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
        changed |= color(ui, "Button hovered", &mut theme.button_hover);
        changed |= color(ui, "Button pressed", &mut theme.button_pressed);
        changed |= color(ui, "Progress fill", &mut theme.progress_fill);
    });
    changed
}

fn style_ui(ui: &mut Ui, style: &mut Style) -> bool {
    let mut changed = color(ui, "Text", &mut style.text_color);
    changed |= color(ui, "Background", &mut style.background);
    changed |= number(ui, "Background opacity", &mut style.background_opacity, 0.0..=1.0, 0.01);
    changed
}

fn number(ui: &mut Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>, speed: f64) -> bool {
    ui.label(label);
    let changed = ui.add(egui::DragValue::new(value).range(range).speed(speed).max_decimals(2)).changed();
    ui.end_row();
    changed
}

fn color(ui: &mut Ui, label: &str, value: &mut [f32; 3]) -> bool {
    ui.label(label);
    let changed = ui.color_edit_button_rgb(value).changed();
    ui.end_row();
    changed
}
