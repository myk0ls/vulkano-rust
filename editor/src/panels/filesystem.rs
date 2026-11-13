//! The FileSystem dock: browses the project's files, opens scenes, and imports models.
//! Right-click a file or folder (or empty space) to create, rename, delete, or show it in
//! the system's file manager.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use egui::{Color32, FontId, Pos2, RichText, Sense, Ui, vec2};

use crate::editor::{Action, Dialog, Editor};
use crate::panels::scripts::FileKind;
use crate::theme;

/// Folders opened at first.
const ROOTS: [&str; 6] = ["scenes", "prefabs", "scripts", "shaders", "data", "assets"];
const ROW_HEIGHT: f32 = 22.0;

/// Dragged from the FileSystem dock: a model, prefab, or other file.
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
    /// All project files with one of these extensions, as asset paths.
    pub fn files_with_extensions(&self, extensions: &[&str]) -> Vec<String> {
        fn walk(entry: &FileEntry, extensions: &[&str], out: &mut Vec<String>) {
            match &entry.children {
                Some(children) => children.iter().for_each(|c| walk(c, extensions, out)),
                None => {
                    if extensions.contains(&extension(&entry.path).as_str()) {
                        out.push(asset_path(&entry.path));
                    }
                }
            }
        }
        let mut out = Vec::new();
        self.roots.iter().for_each(|r| walk(r, extensions, &mut out));
        out.sort();
        out
    }

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

/// Everything in the project folder, except hidden files, build output, and Rust crates
/// (the engine's own source, when the project is the engine's repository).
fn scan_project() -> Vec<FileEntry> {
    let mut entries: Vec<FileEntry> = std::fs::read_dir(".")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| scan(Path::new(&e.file_name())))
        .collect();
    entries.sort_by_key(|e| (e.children.is_none(), e.name.to_lowercase()));
    entries
}

fn scan(path: &Path) -> Option<FileEntry> {
    let name = path.file_name()?.to_string_lossy().into_owned();
    if name.starts_with('.') || name == "target" || (path.is_dir() && path.join("Cargo.toml").is_file()) {
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
    matches!(extension(path).as_str(), "glb" | "gltf" | "blend")
}

fn matches_filter(entry: &FileEntry, filter: &str) -> bool {
    entry.name.to_lowercase().contains(filter)
        || entry.children.as_ref().is_some_and(|c| c.iter().any(|e| matches_filter(e, filter)))
}

fn icon(entry: &FileEntry) -> (&'static str, Color32) {
    if entry.children.is_some() {
        return ("🗀", theme::accent());
    }
    match extension(&entry.path).as_str() {
        "vscene" => ("🎬", theme::play()),
        "rhai" => ("🖹", theme::accent()),
        "glsl" => ("🖹", theme::collider()),
        "glb" | "gltf" | "blend" => ("▣", theme::entity()),
        "hdr" | "png" | "jpg" | "jpeg" => ("🖼", theme::warning()),
        "wav" | "ogg" | "mp3" | "flac" => ("♫", theme::play()),
        _ => ("🗋", theme::text_dim()),
    }
}

/// The last part of an asset path, for showing in the UI.
pub fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

/// A path as scene files should store it: relative, with forward slashes.
fn asset_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

pub fn ui(ui: &mut Ui, editor: &mut Editor) {
    if editor.files.stale {
        editor.files.roots = scan_project();
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
            ui.label(RichText::new("This project is empty. Right-click to add files.").color(theme::text_dim()));
        }
        // The space below the files: right-click to add to the project folder itself.
        let rest = ui.available_size().max(vec2(0.0, 40.0));
        let (_, response) = ui.allocate_exact_size(rest, Sense::click());
        response.context_menu(|ui| context_menu(ui, editor, None));
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
    let model = is_model(&entry.path) || extension(&entry.path) == "vscene";

    let sense = if model { Sense::click_and_drag() } else { Sense::click() };
    let (rect, response) =
        ui.allocate_exact_size(vec2(ui.available_width(), ROW_HEIGHT), sense);

    if ui.is_rect_visible(rect) {
        let selected = editor.files.selected.as_ref() == Some(&entry.path);
        if selected {
            ui.painter().rect_filled(rect, 2.0, theme::selected());
        } else if response.hovered() {
            ui.painter().rect_filled(rect, 2.0, theme::hover());
        }
        let x = rect.left() + 6.0 + depth as f32 * 14.0;
        let y = rect.center().y;
        if is_folder {
            let arrow = if open { "⏷" } else { "⏵" };
            text(ui, Pos2::new(x, y), arrow, theme::text_dim());
        }
        let (glyph, color) = icon(entry);
        text(ui, Pos2::new(x + 14.0, y), glyph, color);
        text(ui, Pos2::new(x + 34.0, y), &entry.name, theme::text());
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
            "rhai" | "glsl" => editor.scripts.open_in_editor(&asset_path(&entry.path), &mut editor.log),
            "glb" | "gltf" | "blend" => editor.actions.push(Action::AddModel {
                path: asset_path(&entry.path),
                position: None,
            }),
            _ => {}
        }
    }
    response.context_menu(|ui| context_menu(ui, editor, Some(entry)));
    if model {
        response.dnd_set_drag_payload(ModelDrag(asset_path(&entry.path)));
        let tip = if is_model(&entry.path) {
            "Double-click or drag into the 3D view to add"
        } else {
            "Double-click to open; drag into the 3D view to place it as a prefab"
        };
        response.clone().on_hover_text(tip);
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

/// What a name typed into the file dialog is for.
#[derive(Clone, Debug, PartialEq)]
pub enum FileOp {
    NewFolder(PathBuf),
    NewFile(PathBuf, NewFile),
    Rename(PathBuf),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NewFile {
    Script,
    Scene,
    Material,
}

impl NewFile {
    fn extension(self) -> &'static str {
        match self {
            NewFile::Script => "rhai",
            NewFile::Scene => "vscene",
            NewFile::Material => "glsl",
        }
    }

    fn label(self) -> &'static str {
        match self {
            NewFile::Script => "Script",
            NewFile::Scene => "Scene",
            NewFile::Material => "Material Shader",
        }
    }
}

impl FileOp {
    pub fn title(&self) -> String {
        match self {
            FileOp::NewFolder(_) => "New Folder".into(),
            FileOp::NewFile(_, kind) => format!("New {}", kind.label()),
            FileOp::Rename(path) => format!("Rename {}", path.file_name().unwrap_or_default().to_string_lossy()),
        }
    }

    /// The folder the name goes in, as shown before the name.
    pub fn folder(&self) -> String {
        let dir = match self {
            FileOp::NewFolder(dir) | FileOp::NewFile(dir, _) => dir.clone(),
            FileOp::Rename(path) => path.parent().map(Path::to_path_buf).unwrap_or_default(),
        };
        let shown = asset_path(&dir);
        if shown.is_empty() || shown == "." { String::new() } else { format!("{shown}/") }
    }

    /// What goes after the name, for new files.
    pub fn suffix(&self) -> String {
        match self {
            FileOp::NewFile(_, kind) => format!(".{}", kind.extension()),
            _ => String::new(),
        }
    }

    /// Do it. Returns the path made, or renamed to.
    pub fn apply(&self, name: &str) -> Result<PathBuf, String> {
        let name = name.trim();
        if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
            return Err("Use a name without slashes that doesn't start with a dot.".into());
        }
        let fail = |what: &str, path: &Path, e: std::io::Error| format!("Could not {what} {}: {e}", path.display());
        match self {
            FileOp::NewFolder(dir) => {
                let path = dir.join(name);
                if path.exists() {
                    return Err(format!("{} already exists.", asset_path(&path)));
                }
                std::fs::create_dir_all(&path).map_err(|e| fail("create", &path, e))?;
                Ok(path)
            }
            FileOp::NewFile(dir, kind) => {
                let path = dir.join(format!("{name}.{}", kind.extension()));
                if path.exists() {
                    return Err(format!("{} already exists.", asset_path(&path)));
                }
                if !dir.as_os_str().is_empty() {
                    std::fs::create_dir_all(dir).map_err(|e| fail("create", dir, e))?;
                }
                match kind {
                    NewFile::Script => std::fs::write(&path, FileKind::Script.template()).map_err(|e| fail("write", &path, e))?,
                    NewFile::Material => std::fs::write(&path, FileKind::Material.template()).map_err(|e| fail("write", &path, e))?,
                    NewFile::Scene => vulkano_engine::scene::scene_file::SceneFile::default().save(&path)?,
                }
                Ok(path)
            }
            FileOp::Rename(path) => {
                let target = path.parent().map(|p| p.join(name)).unwrap_or_else(|| PathBuf::from(name));
                if target.exists() {
                    return Err(format!("{} already exists.", asset_path(&target)));
                }
                std::fs::rename(path, &target).map_err(|e| fail("rename", path, e))?;
                Ok(target)
            }
        }
    }
}

/// The folder new things go in: the folder clicked, or the folder of the file clicked, or
/// the project folder.
fn target_folder(entry: Option<&FileEntry>) -> PathBuf {
    match entry {
        Some(e) if e.children.is_some() => e.path.clone(),
        Some(e) => e.path.parent().map(Path::to_path_buf).unwrap_or_default(),
        None => PathBuf::new(),
    }
}

fn context_menu(ui: &mut Ui, editor: &mut Editor, entry: Option<&FileEntry>) {
    let folder = target_folder(entry);
    let name_dialog = |op: FileOp, name: &str| Dialog::FileName { op, name: name.to_string() };

    if let Some(entry) = entry.filter(|e| e.children.is_none()) {
        let path = asset_path(&entry.path);
        let ext = extension(&entry.path);
        let open_label = match ext.as_str() {
            "vscene" => Some("Open Scene"),
            "rhai" | "glsl" => Some("Open in Code Editor"),
            _ if is_model(&entry.path) => Some("Add to Scene"),
            _ => None,
        };
        if let Some(label) = open_label {
            if ui.button(label).clicked() {
                match ext.as_str() {
                    "vscene" => editor.actions.push(Action::OpenScene(entry.path.clone())),
                    "rhai" | "glsl" => editor.scripts.open_in_editor(&path, &mut editor.log),
                    _ => editor.actions.push(Action::AddModel { path: path.clone(), position: None }),
                }
                ui.close();
            }
        }
        if ui.button("Open with System App").clicked() {
            if let Err(e) = crate::projects::open_with_system(&entry.path) {
                editor.log.error(e);
            }
            ui.close();
        }
    }
    let shown = entry.map_or(PathBuf::from("."), |e| e.path.clone());
    if ui.button(if cfg!(windows) { "Show in Explorer" } else { "Show in File Manager" }).clicked() {
        if let Err(e) = crate::projects::reveal(&shown) {
            editor.log.error(e);
        }
        ui.close();
    }
    ui.separator();
    if ui.button("New Folder…").clicked() {
        editor.dialog = Some(name_dialog(FileOp::NewFolder(folder.clone()), "new_folder"));
        ui.close();
    }
    for (kind, default) in [(NewFile::Scene, "new_scene"), (NewFile::Script, "new_script"), (NewFile::Material, "new_material")] {
        if ui.button(format!("New {}…", kind.label())).clicked() {
            editor.dialog = Some(name_dialog(FileOp::NewFile(folder.clone(), kind), default));
            ui.close();
        }
    }
    if let Some(entry) = entry {
        ui.separator();
        if ui.button("Rename…").clicked() {
            editor.dialog = Some(name_dialog(FileOp::Rename(entry.path.clone()), &entry.name));
            ui.close();
        }
        if ui.button("Copy Path").clicked() {
            ui.ctx().copy_text(asset_path(&entry.path));
            ui.close();
        }
        if ui.button(RichText::new("Delete…").color(theme::error())).clicked() {
            editor.dialog = Some(Dialog::DeleteFile { path: entry.path.clone() });
            ui.close();
        }
    }
    ui.separator();
    if ui.button("Rescan Files").clicked() {
        editor.files.refresh();
        ui.close();
    }
}

/// A path as the editor and scenes write it: relative, with forward slashes.
pub fn display_path(path: &Path) -> String {
    asset_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_ops_create_rename_and_refuse_clashes() {
        let dir = std::env::temp_dir().join(format!("fsops-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let script = FileOp::NewFile(dir.clone(), NewFile::Script).apply("hello").unwrap();
        assert!(script.ends_with("hello.rhai") && script.is_file());
        assert!(FileOp::NewFile(dir.clone(), NewFile::Script).apply("hello").is_err());
        let scene = FileOp::NewFile(dir.clone(), NewFile::Scene).apply("level").unwrap();
        assert!(vulkano_engine::scene::scene_file::SceneFile::load(&scene).is_ok());
        let folder = FileOp::NewFolder(dir.clone()).apply("models").unwrap();
        assert!(folder.is_dir());
        let renamed = FileOp::Rename(script.clone()).apply("greet.rhai").unwrap();
        assert!(renamed.is_file() && !script.exists());
        assert!(FileOp::NewFolder(dir.clone()).apply("a/b").is_err());
        assert!(FileOp::NewFolder(dir.clone()).apply(".hidden").is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
