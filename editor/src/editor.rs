//! Editor state, the main layout, menus, dialogs, and user actions.

use std::collections::HashMap;
use std::path::PathBuf;

use egui::{Align, Button, Key, KeyboardShortcut, Layout, Modifiers, RichText, Ui};
use egui_dock::{DockArea, DockState, NodeIndex, TabViewer};
use vulkano_engine::scene::scene_file::{NodeId, NodeKind, NodeTransform, SceneFile};

use crate::document::Document;
use crate::log::Log;
use crate::panels::filesystem::FileBrowser;
use crate::panels::scene_tree::TreeState;
use crate::panels::viewport::ViewportState;
use crate::panels::{filesystem, inspector, output, scene_tree, viewport};
use crate::runner::GameRunner;
use crate::sync::ModelInfo;
use crate::theme;

const SCENES_DIR: &str = "scenes";

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tab {
    Viewport,
    Scene,
    FileSystem,
    Inspector,
    Output,
}

#[derive(Clone, Debug)]
pub enum Action {
    NewScene,
    OpenScene(PathBuf),
    Save,
    SaveAs(PathBuf),
    Quit,
    Play,
    Stop,
    Undo,
    Redo,
    DeleteSelected,
    DuplicateSelected,
    AddModel {
        path: String,
        position: Option<[f32; 3]>,
    },
}

pub enum Dialog {
    CreateNode { parent: Option<NodeId>, search: String },
    OpenScene { files: Vec<PathBuf> },
    SaveAs { name: String, then: Option<Action> },
    Unsaved { then: Action },
    Shortcuts,
    About,
}

pub struct Editor {
    pub doc: Document,
    pub log: Log,
    pub runner: GameRunner,
    pub viewport: ViewportState,
    pub tree: TreeState,
    pub files: FileBrowser,
    pub models: HashMap<String, ModelInfo>,
    pub dialog: Option<Dialog>,
    pub actions: Vec<Action>,
    /// A blocking operation is about to run; the UI shows this while it does.
    pub busy: Option<String>,
    pub quit: bool,
    pub frame_ms: f32,
    pub output_autoscroll: bool,
    pub reload_requested: bool,
    pub frame_on_load: bool,
    dock: Option<DockState<Tab>>,
}

fn default_layout() -> DockState<Tab> {
    let mut state = DockState::new(vec![Tab::Viewport]);
    let surface = state.main_surface_mut();
    let [center, _inspector] = surface.split_right(NodeIndex::root(), 0.78, vec![Tab::Inspector]);
    let [center, left] = surface.split_left(center, 0.24, vec![Tab::Scene]);
    surface.split_below(left, 0.55, vec![Tab::FileSystem]);
    surface.split_below(center, 0.78, vec![Tab::Output]);
    state
}

impl Editor {
    pub fn new() -> Self {
        let mut log = Log::default();
        log.info("Vulkano Editor ready. Open a scene from the FileSystem dock, or add a node to begin.");
        Self {
            doc: Document::new(),
            log,
            runner: GameRunner::new(),
            viewport: ViewportState::new(),
            tree: TreeState::default(),
            files: FileBrowser::new(),
            models: HashMap::new(),
            dialog: None,
            actions: Vec::new(),
            busy: None,
            quit: false,
            frame_ms: 0.0,
            output_autoscroll: true,
            reload_requested: false,
            frame_on_load: false,
            dock: Some(default_layout()),
        }
    }

    pub fn window_title(&self) -> String {
        let dirty = if self.doc.is_dirty() { "(*) " } else { "" };
        format!("{dirty}{} - Vulkano Editor", self.doc.title())
    }

    pub fn ui(&mut self, root: &mut Ui) {
        let ctx = root.ctx().clone();

        egui::Panel::top("menu_bar")
            .frame(egui::Frame::new().fill(theme::DARK).inner_margin(egui::Margin::symmetric(6, 4)))
            .show(root, |ui| self.top_bar(ui));

        let mut dock = self.dock.take().unwrap_or_else(default_layout);
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::DARK))
            .show(root, |ui| {
                DockArea::new(&mut dock)
                    .style(theme::dock_style(&ctx))
                    .show_close_buttons(false)
                    .show_add_buttons(false)
                    .show_leaf_close_all_buttons(false)
                    .show_leaf_collapse_buttons(false)
                    .show_inside(ui, &mut Tabs { editor: self });
            });
        // The layout can be reset from a menu while the dock is taken.
        if self.dock.is_none() {
            self.dock = Some(dock);
        }

        self.dialogs(&ctx);
        self.shortcuts(&ctx);
        self.busy_overlay(&ctx);
        self.apply_actions();

        // A drag or an in-progress text edit counts as one undo step.
        let interacting = ctx.input(|i| i.pointer.any_down()) || ctx.egui_wants_keyboard_input();
        self.doc.commit_changes(interacting);
    }

    pub fn is_blocked_by_dialog(&self) -> bool {
        self.dialog.is_some()
    }

    fn top_bar(&mut self, ui: &mut Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("Scene", |ui| {
                self.menu_item(ui, "New Scene", "Ctrl+N", true, Action::NewScene);
                if ui.add(Button::new("Open Scene…").shortcut_text("Ctrl+O")).clicked() {
                    self.open_dialog();
                    ui.close();
                }
                ui.separator();
                self.menu_item(ui, "Save Scene", "Ctrl+S", true, Action::Save);
                if ui.add(Button::new("Save Scene As…").shortcut_text("Ctrl+Shift+S")).clicked() {
                    self.save_as_dialog(None);
                    ui.close();
                }
                ui.separator();
                self.menu_item(ui, "Quit", "Ctrl+Q", true, Action::Quit);
            });
            ui.menu_button("Edit", |ui| {
                self.menu_item(ui, "Undo", "Ctrl+Z", self.doc.can_undo(), Action::Undo);
                self.menu_item(ui, "Redo", "Ctrl+Y", self.doc.can_redo(), Action::Redo);
                ui.separator();
                let selected = self.doc.selection.is_some();
                self.menu_item(ui, "Duplicate", "Ctrl+D", selected, Action::DuplicateSelected);
                self.menu_item(ui, "Delete", "Del", selected, Action::DeleteSelected);
            });
            ui.menu_button("View", |ui| {
                ui.checkbox(&mut self.viewport.show_grid, "Grid");
                ui.checkbox(&mut self.viewport.show_icons, "Node Icons");
                ui.separator();
                if ui.button("Reset Dock Layout").clicked() {
                    self.dock = Some(default_layout());
                    ui.close();
                }
            });
            ui.menu_button("Help", |ui| {
                if ui.button("Controls & Shortcuts").clicked() {
                    self.dialog = Some(Dialog::Shortcuts);
                    ui.close();
                }
                if ui.button("About").clicked() {
                    self.dialog = Some(Dialog::About);
                    ui.close();
                }
            });

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let running = self.runner.is_running();
                let stop = ui
                    .add_enabled(running, Button::new("⏹").min_size(egui::vec2(28.0, 22.0)))
                    .on_hover_text("Stop the running game (F8)");
                if stop.clicked() {
                    self.actions.push(Action::Stop);
                }
                let play_color = if running { theme::TEXT_DIM } else { theme::PLAY };
                let play = ui
                    .add(Button::new(RichText::new("▶").color(play_color)).min_size(egui::vec2(28.0, 22.0)))
                    .on_hover_text("Save and run this scene in the game (F5)");
                if play.clicked() {
                    self.actions.push(Action::Play);
                }
                ui.add_space(8.0);
                let dirty = if self.doc.is_dirty() { " (*)" } else { "" };
                ui.label(RichText::new(format!("{}{dirty}", self.doc.title())).color(theme::TEXT_DIM));
            });
        });
    }

    fn menu_item(&mut self, ui: &mut Ui, label: &str, shortcut: &str, enabled: bool, action: Action) {
        let button = Button::new(label).shortcut_text(shortcut);
        if ui.add_enabled(enabled, button).clicked() {
            self.actions.push(action);
            ui.close();
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        if self.dialog.is_some() {
            return;
        }
        let ctrl = Modifiers::COMMAND;
        let ctrl_shift = Modifiers::COMMAND | Modifiers::SHIFT;
        let pressed = |modifiers, key| {
            ctx.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(modifiers, key)))
        };

        if pressed(ctrl_shift, Key::S) {
            self.save_as_dialog(None);
        }
        if pressed(ctrl_shift, Key::Z) {
            self.actions.push(Action::Redo);
        }
        if pressed(ctrl, Key::S) {
            self.actions.push(Action::Save);
        }
        if pressed(ctrl, Key::N) {
            self.actions.push(Action::NewScene);
        }
        if pressed(ctrl, Key::O) {
            self.open_dialog();
        }
        if pressed(ctrl, Key::Q) {
            self.actions.push(Action::Quit);
        }
        if pressed(Modifiers::NONE, Key::F5) {
            self.actions.push(Action::Play);
        }
        if pressed(Modifiers::NONE, Key::F8) {
            self.actions.push(Action::Stop);
        }

        // These mean something else while typing in a field.
        if !ctx.egui_wants_keyboard_input() {
            if pressed(ctrl, Key::Z) {
                self.actions.push(Action::Undo);
            }
            if pressed(ctrl, Key::Y) {
                self.actions.push(Action::Redo);
            }
            if pressed(ctrl, Key::D) {
                self.actions.push(Action::DuplicateSelected);
            }
            if pressed(ctrl, Key::A) {
                self.dialog = Some(Dialog::CreateNode {
                    parent: self.doc.selection,
                    search: String::new(),
                });
            }
            if pressed(Modifiers::NONE, Key::Delete) {
                self.actions.push(Action::DeleteSelected);
            }
            if pressed(Modifiers::NONE, Key::F2) {
                if let Some(node) = self.doc.selection.and_then(|id| self.doc.node(id)) {
                    self.tree.begin_rename(node.id, node.name.clone());
                }
            }
        }
    }

    pub fn open_dialog(&mut self) {
        let files = filesystem::list_scene_files(SCENES_DIR);
        self.dialog = Some(Dialog::OpenScene { files });
    }

    fn save_as_dialog(&mut self, then: Option<Action>) {
        let name = self
            .doc
            .path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "new_scene".to_string());
        self.dialog = Some(Dialog::SaveAs { name, then });
    }

    fn apply_actions(&mut self) {
        for action in std::mem::take(&mut self.actions) {
            self.apply(action, false);
        }
    }

    /// Carry out an action. `discard` means the user already agreed to lose changes.
    fn apply(&mut self, action: Action, discard: bool) {
        let replaces_doc = matches!(action, Action::NewScene | Action::OpenScene(_) | Action::Quit);
        if replaces_doc && !discard && self.doc.is_dirty() {
            self.dialog = Some(Dialog::Unsaved { then: action });
            return;
        }

        match action {
            Action::NewScene => {
                self.doc = Document::new();
                self.log.info("New scene.");
            }
            Action::OpenScene(path) => match SceneFile::load(&path) {
                Ok(scene) => {
                    self.log.info(format!("Opened {}.", path.display()));
                    self.doc = Document::from_scene(scene, Some(path));
                    self.tree.reset();
                    self.reload_requested = true;
                    self.frame_on_load = true;
                }
                Err(e) => self.log.error(e),
            },
            Action::Save => match self.doc.path.clone() {
                Some(path) => self.save_to(path),
                None => self.save_as_dialog(None),
            },
            Action::SaveAs(path) => self.save_to(path),
            Action::Quit => self.quit = true,
            Action::Play => match self.doc.path.clone() {
                Some(path) => {
                    self.save_to(path.clone());
                    if !self.doc.is_dirty() {
                        self.runner.start(&path, &mut self.log);
                    }
                }
                None => self.save_as_dialog(Some(Action::Play)),
            },
            Action::Stop => self.runner.stop(&mut self.log),
            Action::Undo => self.doc.undo(),
            Action::Redo => self.doc.redo(),
            Action::DeleteSelected => {
                if let Some(id) = self.doc.selection {
                    self.doc.remove_subtree(id);
                }
            }
            Action::DuplicateSelected => {
                if let Some(id) = self.doc.selection {
                    self.doc.duplicate(id);
                }
            }
            Action::AddModel { path, position } => {
                let name = std::path::Path::new(&path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "Model".to_string());
                let transform = NodeTransform {
                    position: position.unwrap_or([0.0; 3]),
                    ..Default::default()
                };
                self.doc.add_node_at(
                    None,
                    &name,
                    NodeKind::Model { path, animation: None },
                    transform,
                );
            }
        }
    }

    fn save_to(&mut self, path: PathBuf) {
        match self.doc.save_to(path.clone()) {
            Ok(()) => {
                self.log.info(format!("Saved {}.", path.display()));
                self.files.refresh();
            }
            Err(e) => self.log.error(e),
        }
    }

    fn busy_overlay(&self, ctx: &egui::Context) {
        let Some(message) = &self.busy else {
            return;
        };
        egui::Modal::new(egui::Id::new("busy")).show(ctx, |ui| {
            ui.set_min_width(260.0);
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(message);
            });
        });
    }

    fn dialogs(&mut self, ctx: &egui::Context) {
        let Some(mut dialog) = self.dialog.take() else {
            return;
        };
        let keep = match &mut dialog {
            Dialog::CreateNode { parent, search } => self.create_node_dialog(ctx, *parent, search),
            Dialog::OpenScene { files } => self.open_scene_dialog(ctx, files),
            Dialog::SaveAs { name, then } => self.save_as_ui(ctx, name, then),
            Dialog::Unsaved { then } => self.unsaved_dialog(ctx, then),
            Dialog::Shortcuts => info_dialog(ctx, "Controls & Shortcuts", shortcuts_ui),
            Dialog::About => info_dialog(ctx, "About", |ui| {
                ui.label("Vulkano Editor");
                ui.label(
                    RichText::new("A scene editor for the vulkano-engine, in the style of Godot.")
                        .color(theme::TEXT_DIM),
                );
            }),
        };
        // A dialog may have opened another one (for example Unsaved -> Save As).
        if keep && self.dialog.is_none() {
            self.dialog = Some(dialog);
        }
    }

    fn create_node_dialog(
        &mut self,
        ctx: &egui::Context,
        parent: Option<NodeId>,
        search: &mut String,
    ) -> bool {
        let mut keep = true;
        let modal = egui::Modal::new(egui::Id::new("create_node")).show(ctx, |ui| {
            ui.set_width(340.0);
            ui.heading("Create New Node");
            ui.add_space(4.0);
            let field = ui.add(egui::TextEdit::singleline(search).hint_text("Search…").desired_width(f32::INFINITY));
            field.request_focus();
            ui.add_space(4.0);

            let query = search.to_lowercase();
            let mut first = None;
            let mut choice = None;
            for (index, (label, description)) in NODE_TYPES.iter().enumerate() {
                if !label.to_lowercase().contains(&query) {
                    continue;
                }
                first.get_or_insert(index);
                let row = ui
                    .add(Button::selectable(false, format!("{}  {label}", scene_tree::type_icon(index))).min_size(egui::vec2(ui.available_width(), 24.0)))
                    .on_hover_text(*description);
                if row.clicked() {
                    choice = Some(index);
                }
            }
            ui.add_space(4.0);
            if ui.input(|i| i.key_pressed(Key::Enter)) {
                choice = first;
            }
            if let Some(index) = choice {
                self.doc.add_node(parent, NODE_TYPES[index].0, node_kind(index));
                keep = false;
            }
            if ui.button("Cancel").clicked() {
                keep = false;
            }
        });
        keep && !modal.should_close()
    }

    fn open_scene_dialog(&mut self, ctx: &egui::Context, files: &[PathBuf]) -> bool {
        let mut keep = true;
        let modal = egui::Modal::new(egui::Id::new("open_scene")).show(ctx, |ui| {
            ui.set_width(420.0);
            ui.heading("Open Scene");
            ui.add_space(4.0);
            if files.is_empty() {
                ui.label(RichText::new(format!("No scenes in {SCENES_DIR}/ yet.")).color(theme::TEXT_DIM));
            }
            egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                for file in files {
                    let row = Button::selectable(false, format!("🎬  {}", file.display()))
                        .min_size(egui::vec2(ui.available_width(), 24.0));
                    if ui.add(row).clicked() {
                        self.actions.push(Action::OpenScene(file.clone()));
                        keep = false;
                    }
                }
            });
            ui.add_space(4.0);
            if ui.button("Cancel").clicked() {
                keep = false;
            }
        });
        keep && !modal.should_close()
    }

    fn save_as_ui(&mut self, ctx: &egui::Context, name: &mut String, then: &mut Option<Action>) -> bool {
        let mut keep = true;
        let modal = egui::Modal::new(egui::Id::new("save_as")).show(ctx, |ui| {
            ui.set_width(380.0);
            ui.heading("Save Scene As");
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{SCENES_DIR}/")).color(theme::TEXT_DIM));
                ui.add(egui::TextEdit::singleline(name).desired_width(220.0)).request_focus();
                ui.label(RichText::new(".vscene").color(theme::TEXT_DIM));
            });
            let cleaned = name.trim();
            let valid = !cleaned.is_empty() && !cleaned.contains(['/', '\\']) && !cleaned.starts_with('.');
            let path = PathBuf::from(SCENES_DIR).join(format!("{cleaned}.vscene"));
            let exists = valid && path.exists() && self.doc.path.as_ref() != Some(&path);
            if exists {
                ui.label(RichText::new("This will replace the existing file.").color(theme::WARNING));
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let enter = ui.input(|i| i.key_pressed(Key::Enter));
                if (ui.add_enabled(valid, Button::new("Save")).clicked() || enter) && valid {
                    self.actions.push(Action::SaveAs(path.clone()));
                    if let Some(next) = then.take() {
                        self.actions.push(next);
                    }
                    keep = false;
                }
                if ui.button("Cancel").clicked() {
                    keep = false;
                }
            });
        });
        keep && !modal.should_close()
    }

    fn unsaved_dialog(&mut self, ctx: &egui::Context, then: &Action) -> bool {
        let mut keep = true;
        let modal = egui::Modal::new(egui::Id::new("unsaved")).show(ctx, |ui| {
            ui.set_width(380.0);
            ui.heading("Unsaved Changes");
            ui.add_space(4.0);
            ui.label(format!("Save changes to {} before continuing?", self.doc.title()));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    match self.doc.path.clone() {
                        Some(path) => {
                            self.save_to(path);
                            if !self.doc.is_dirty() {
                                self.apply(then.clone(), true);
                            }
                        }
                        None => self.save_as_dialog(Some(then.clone())),
                    }
                    keep = false;
                }
                if ui.button("Don't Save").clicked() {
                    self.apply(then.clone(), true);
                    keep = false;
                }
                if ui.button("Cancel").clicked() {
                    keep = false;
                }
            });
        });
        keep && !modal.should_close()
    }
}

/// The node types offered by the Create New Node dialog.
pub const NODE_TYPES: [(&str, &str); 2] = [
    ("Node3D", "An empty transform, for grouping or marking a place. Name one PlayerSpawn to set where the game starts."),
    ("PointLight", "A light that shines in all directions."),
];

fn node_kind(index: usize) -> NodeKind {
    match index {
        0 => NodeKind::Empty,
        _ => NodeKind::PointLight {
            color: [1.0, 1.0, 1.0],
            intensity: 5.0,
            radius: 5.0,
        },
    }
}

pub fn add_node_of_type(doc: &mut Document, parent: Option<NodeId>, index: usize) {
    doc.add_node(parent, NODE_TYPES[index].0, node_kind(index));
}

fn info_dialog(ctx: &egui::Context, title: &str, contents: impl FnOnce(&mut Ui)) -> bool {
    let mut keep = true;
    let modal = egui::Modal::new(egui::Id::new(title)).show(ctx, |ui| {
        ui.set_width(460.0);
        ui.heading(title);
        ui.add_space(6.0);
        contents(ui);
        ui.add_space(8.0);
        if ui.button("Close").clicked() {
            keep = false;
        }
    });
    keep && !modal.should_close()
}

fn shortcuts_ui(ui: &mut Ui) {
    let rows = [
        ("Viewport", ""),
        ("Right mouse + WASD / Q E", "Fly; wheel changes speed; Shift is faster"),
        ("Middle mouse", "Orbit"),
        ("Shift + middle mouse", "Pan"),
        ("Wheel", "Zoom"),
        ("Left click", "Select"),
        ("Q  W  E  R", "Select, Move, Rotate, Scale tool"),
        ("F", "Focus the selection"),
        ("Ctrl while using a gizmo", "Snap"),
        ("", ""),
        ("Scene", ""),
        ("Ctrl+A", "Add a node"),
        ("Ctrl+D  /  Del  /  F2", "Duplicate, delete, rename"),
        ("Ctrl+Z  /  Ctrl+Y", "Undo, redo"),
        ("Ctrl+S  /  Ctrl+Shift+S", "Save, save as"),
        ("F5  /  F8", "Run the scene, stop the game"),
    ];
    egui::Grid::new("shortcuts").num_columns(2).spacing([24.0, 4.0]).show(ui, |ui| {
        for (keys, description) in rows {
            if description.is_empty() {
                ui.label(RichText::new(keys).strong().color(theme::ACCENT));
            } else {
                ui.label(keys);
                ui.label(RichText::new(description).color(theme::TEXT_DIM));
            }
            ui.end_row();
        }
    });
}

struct Tabs<'a> {
    editor: &'a mut Editor,
}

impl TabViewer for Tabs<'_> {
    type Tab = Tab;

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        egui::Id::new(*tab as u8 as u64 + 1000)
    }

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        match tab {
            Tab::Viewport => "3D",
            Tab::Scene => "Scene",
            Tab::FileSystem => "FileSystem",
            Tab::Inspector => "Inspector",
            Tab::Output => "Output",
        }
        .into()
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Tab) {
        match tab {
            Tab::Viewport => viewport::ui(ui, self.editor),
            Tab::Scene => scene_tree::ui(ui, self.editor),
            Tab::FileSystem => filesystem::ui(ui, self.editor),
            Tab::Inspector => inspector::ui(ui, self.editor),
            Tab::Output => output::ui(ui, self.editor),
        }
    }

    fn is_closeable(&self, _tab: &Tab) -> bool {
        false
    }

    fn allowed_in_windows(&self, tab: &mut Tab) -> bool {
        // The viewport's render target assumes it is part of the main layout.
        *tab != Tab::Viewport
    }

    fn clear_background(&self, tab: &Tab) -> bool {
        *tab != Tab::Viewport
    }
}

