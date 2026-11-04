//! The FileSystem dock: browses project files, opens scenes, and imports models.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use egui::{Color32, FontId, Pos2, RichText, Sense, Ui, vec2};

use crate::editor::{Action, Editor};
use crate::theme;

const ROOTS: [&str; 2] = ["scenes", "data"];
const ROW_HEIGHT: f32 = 22.0;

/// Dragged from the FileSystem dock onto the viewport to add a model.
#[derive(Clone)]
pub struct ModelDrag(pub String);

struct FileEntry {
    name: String,
    path: PathBuf,
    children: Option<Vec<FileEntry>>,
}

pub struct FileBrowser {
    roots: Vec<FileEntry>,
    expanded: HashSet<PathBuf>,
    selected: Option<PathBuf>,
    filter: String,
    stale: bool,
}

impl FileBrowser {
    pub fn new() -> Self {
        Self {
            roots: Vec::new(),
            expanded: ROOTS.iter().map(PathBuf::from).collect(),
            selected: None,
            filter: String::new(),
            stale: true,
        }
    }

    pub fn refresh(&mut self) {
        self.stale = true;
    }
}

fn scan(path: &Path) -> Option<FileEntry> {
    let name = path.file_name()?.to_string_lossy().into_owned();
    if name.starts_with('.') {
        return None;
    }
    let children = path.is_dir().then(|| {
        let mut entries: Vec<FileEntry> = std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| scan(&e.path()))
            .collect();
        // Folders first, like Godot.
        entries.sort_by_key(|e| (e.children.is_none(), e.name.to_lowercase()));
        entries
    });
    Some(FileEntry { name, path: path.to_path_buf(), children })
}

pub fn list_scene_files(dir: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| extension(p) == "vscene")
        .collect();
    files.sort();
    files
}

fn extension(path: &Path) -> String {
    path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

fn is_model(path: &Path) -> bool {
    matches!(extension(path).as_str(), "glb" | "gltf")
}

fn matches_filter(entry: &FileEntry, filter: &str) -> bool {
    entry.name.to_lowercase().contains(filter)
        || entry.children.as_ref().is_some_and(|c| c.iter().any(|e| matches_filter(e, filter)))
}

fn icon(entry: &FileEntry) -> (&'static str, Color32) {
    if entry.children.is_some() {
        return ("🗀", theme::ACCENT);
    }
    match extension(&entry.path).as_str() {
        "vscene" => ("🎬", theme::PLAY),
        "glb" | "gltf" => ("▣", theme::NODE_3D),
        "hdr" | "png" | "jpg" | "jpeg" => ("🖼", theme::WARNING),
        _ => ("🗋", theme::TEXT_DIM),
    }
}

/// A path as scene files should store it: relative, with forward slashes.
fn asset_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

pub fn ui(ui: &mut Ui, editor: &mut Editor) {
    if editor.files.stale {
        editor.files.roots = ROOTS.iter().filter_map(|r| scan(Path::new(r))).collect();
        editor.files.stale = false;
    }

    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut editor.files.filter)
                .hint_text("Filter Files")
                .desired_width(ui.available_width() - 34.0),
        );
        if ui.button("⟳").on_hover_text("Rescan files").clicked() {
            editor.files.refresh();
        }
    });
    ui.add_space(2.0);

    let filter = editor.files.filter.to_lowercase();
    let roots = std::mem::take(&mut editor.files.roots);
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        for entry in &roots {
            row(ui, editor, entry, 0, &filter);
        }
        if roots.is_empty() {
            ui.label(RichText::new("No scenes/ or data/ folder here.").color(theme::TEXT_DIM));
        }
    });
    if editor.files.roots.is_empty() {
        editor.files.roots = roots;
    }
}

fn row(ui: &mut Ui, editor: &mut Editor, entry: &FileEntry, depth: usize, filter: &str) {
    if !filter.is_empty() && !matches_filter(entry, filter) {
        return;
    }
    let is_folder = entry.children.is_some();
    let open = !filter.is_empty() || editor.files.expanded.contains(&entry.path);
    let model = is_model(&entry.path);

    let sense = if model { Sense::click_and_drag() } else { Sense::click() };
    let (rect, response) =
        ui.allocate_exact_size(vec2(ui.available_width(), ROW_HEIGHT), sense);

    if ui.is_rect_visible(rect) {
        let selected = editor.files.selected.as_ref() == Some(&entry.path);
        if selected {
            ui.painter().rect_filled(rect, 2.0, theme::SELECTED);
        } else if response.hovered() {
            ui.painter().rect_filled(rect, 2.0, theme::HOVER);
        }
        let x = rect.left() + 6.0 + depth as f32 * 14.0;
        let y = rect.center().y;
        if is_folder {
            let arrow = if open { "⏷" } else { "⏵" };
            text(ui, Pos2::new(x, y), arrow, theme::TEXT_DIM);
        }
        let (glyph, color) = icon(entry);
        text(ui, Pos2::new(x + 14.0, y), glyph, color);
        text(ui, Pos2::new(x + 34.0, y), &entry.name, theme::TEXT);
    }

    if response.clicked() {
        editor.files.selected = Some(entry.path.clone());
        if is_folder && !editor.files.expanded.remove(&entry.path) {
            editor.files.expanded.insert(entry.path.clone());
        }
    }
    if response.double_clicked() {
        match extension(&entry.path).as_str() {
            "vscene" => editor.actions.push(Action::OpenScene(entry.path.clone())),
            "glb" | "gltf" => editor.actions.push(Action::AddModel {
                path: asset_path(&entry.path),
                position: None,
            }),
            _ => {}
        }
    }
    if model {
        response.dnd_set_drag_payload(ModelDrag(asset_path(&entry.path)));
        response.clone().on_hover_text("Double-click or drag into the 3D view to add");
    }

    if open {
        if let Some(children) = &entry.children {
            for child in children {
                self::row(ui, editor, child, depth + 1, filter);
            }
        }
    }
}

fn text(ui: &Ui, pos: Pos2, text: &str, color: Color32) {
    ui.painter().text(
        pos,
        egui::Align2::LEFT_CENTER,
        text,
        FontId::proportional(13.0),
        color,
    );
}
