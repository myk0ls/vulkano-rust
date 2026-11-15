//! UI themes: a `.vtheme` file giving UI elements a font, shapes and colours, so a game's
//! menus and HUD look alike without styling each element. An element uses one through its
//! `theme` field; with `custom_colors` it keeps its own colours and takes only the font
//! and shapes.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use egui::{Color32, FontFamily, FontId};
use serde::{Deserialize, Serialize};

/// Colours for one kind of element.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Style {
    pub text_color: [f32; 3],
    pub background: [f32; 3],
    pub background_opacity: f32,
}

impl Default for Style {
    fn default() -> Self {
        Self { text_color: [1.0, 1.0, 1.0], background: [0.1, 0.1, 0.12], background_opacity: 0.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UiTheme {
    /// A .ttf or .otf file. Empty uses the built-in font.
    pub font: String,
    /// Multiplies every element's font size.
    pub font_scale: f32,
    pub corner_radius: f32,
    pub border_width: f32,
    pub border_color: [f32; 3],
    pub border_opacity: f32,
    /// A soft shadow under label text, which keeps it readable over the scene.
    pub text_shadow: bool,
    pub label: Style,
    pub button: Style,
    /// A button's background under the pointer, and while held down.
    pub button_hover: [f32; 3],
    pub button_pressed: [f32; 3],
    pub panel: Style,
    /// A ProgressBar's empty part; `progress_fill` is the full part.
    pub progress: Style,
    pub progress_fill: [f32; 3],
}

impl Default for UiTheme {
    fn default() -> Self {
        Self {
            font: String::new(),
            font_scale: 1.0,
            corner_radius: 6.0,
            border_width: 1.0,
            border_color: [1.0, 1.0, 1.0],
            border_opacity: 0.25,
            text_shadow: true,
            label: Style::default(),
            button: Style { background: [0.18, 0.2, 0.26], background_opacity: 0.9, ..Default::default() },
            button_hover: [0.26, 0.3, 0.4],
            button_pressed: [0.12, 0.13, 0.17],
            panel: Style { background_opacity: 0.75, ..Default::default() },
            progress: Style { background: [0.05, 0.05, 0.06], background_opacity: 0.6, ..Default::default() },
            progress_fill: [0.35, 0.75, 0.35],
        }
    }
}

impl UiTheme {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("could not read {path}: {e}"))?;
        ron::from_str(&text).map_err(|e| format!("could not parse {path}: {e}"))
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let text = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::new())
            .map_err(|e| format!("could not write the theme: {e}"))?;
        std::fs::write(path, text).map_err(|e| format!("could not write {path}: {e}"))
    }

    /// The colours for an element kind.
    pub fn style(&self, kind: &str) -> &Style {
        match kind {
            "Button" => &self.button,
            "Panel" => &self.panel,
            "ProgressBar" => &self.progress,
            _ => &self.label,
        }
    }
}

struct Cached {
    modified: Option<SystemTime>,
    theme: Option<Arc<UiTheme>>,
}

static THEMES: Mutex<Option<HashMap<String, Cached>>> = Mutex::new(None);

/// The theme in a file, read again when the file changes so edits show at once. A broken
/// file is reported once and treated as no theme.
pub fn get(path: &str) -> Option<Arc<UiTheme>> {
    if path.is_empty() {
        return None;
    }
    let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    let mut themes = THEMES.lock().unwrap();
    let themes = themes.get_or_insert_with(HashMap::new);
    if let Some(cached) = themes.get(path) {
        if cached.modified == modified {
            return cached.theme.clone();
        }
    }
    let theme = match UiTheme::load(path) {
        Ok(theme) => Some(Arc::new(theme)),
        Err(e) => {
            eprintln!("[ui] {e}");
            None
        }
    };
    themes.insert(path.to_string(), Cached { modified, theme: theme.clone() });
    theme
}

/// Save a theme and use it at once, without waiting to notice the file changed.
pub fn store(path: &str, theme: &UiTheme) -> Result<(), String> {
    theme.save(path)?;
    let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    let mut themes = THEMES.lock().unwrap();
    themes
        .get_or_insert_with(HashMap::new)
        .insert(path.to_string(), Cached { modified, theme: Some(Arc::new(theme.clone())) });
    Ok(())
}

/// The font for a size, loading the theme's font file into egui the first time. A newly
/// added font is usable from the next frame; until then the built-in one is used.
pub fn font(ctx: &egui::Context, theme: Option<&UiTheme>, size: f32) -> FontId {
    let Some(path) = theme.map(|t| t.font.as_str()).filter(|f| !f.is_empty()) else {
        return FontId::proportional(size);
    };
    let family = FontFamily::Name(path.into());
    if ctx.fonts(|fonts| fonts.families().contains(&family)) {
        return FontId::new(size, family);
    }
    if Path::new(path).is_file() {
        match std::fs::read(path) {
            Ok(bytes) => ctx.add_font(egui::epaint::text::FontInsert::new(
                path,
                egui::FontData::from_owned(bytes),
                vec![egui::epaint::text::InsertFontFamily {
                    family,
                    priority: egui::epaint::text::FontPriority::Highest,
                }],
            )),
            Err(e) => eprintln!("[ui] could not read the font {path}: {e}"),
        }
    }
    FontId::proportional(size)
}

pub fn color(rgb: [f32; 3], alpha: f32) -> Color32 {
    let [r, g, b] = rgb.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
    Color32::from_rgba_unmultiplied(r, g, b, (alpha.clamp(0.0, 1.0) * 255.0).round() as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn themes_round_trip_and_fill_in_missing_values() {
        let theme = UiTheme { font: "assets/fonts/x.ttf".into(), corner_radius: 12.0, ..Default::default() };
        let path = std::env::temp_dir().join(format!("theme-test-{}.vtheme", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        theme.save(&path).unwrap();
        assert_eq!(UiTheme::load(&path).unwrap(), theme);
        std::fs::remove_file(&path).unwrap();

        let partial: UiTheme = ron::from_str("(corner_radius: 2.0)").unwrap();
        assert_eq!(partial.corner_radius, 2.0);
        assert_eq!(partial.button_hover, UiTheme::default().button_hover);
    }
}
