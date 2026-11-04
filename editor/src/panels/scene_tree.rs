//! The Scene dock: the node tree, with selection, renaming, and drag to reparent.

use std::collections::HashSet;

use egui::{Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Ui, vec2};
use vulkano_engine::scene::scene_file::{NodeId, NodeKind, SceneNode};

use crate::document::{Document, DropPlace};
use crate::editor::{self, Action, Dialog, Editor};
use crate::theme;

const ROW_HEIGHT: f32 = 22.0;
const INDENT: f32 = 16.0;

#[derive(Default)]
pub struct TreeState {
    collapsed: HashSet<NodeId>,
    rename: Option<(NodeId, String)>,
    rename_needs_focus: bool,
    filter: String,
}

impl TreeState {
    pub fn begin_rename(&mut self, id: NodeId, name: String) {
        self.rename = Some((id, name));
        self.rename_needs_focus = true;
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

pub fn kind_icon(kind: &NodeKind) -> (&'static str, Color32) {
    match kind {
        NodeKind::Empty => ("○", theme::NODE_3D),
        NodeKind::Model { .. } => ("▣", theme::NODE_3D),
        NodeKind::PointLight { .. } => ("💡", theme::WARNING),
    }
}

pub fn type_icon(index: usize) -> &'static str {
    if index == 0 { "○" } else { "💡" }
}

enum TreeAction {
    Select(NodeId),
    ToggleCollapse(NodeId),
    ToggleVisible(NodeId),
    Move(NodeId, DropPlace),
    Rename(NodeId, String),
}

pub fn ui(ui: &mut Ui, editor: &mut Editor) {
    ui.horizontal(|ui| {
        if ui.button("✚").on_hover_text("Add child node (Ctrl+A)").clicked() {
            editor.dialog = Some(Dialog::CreateNode {
                parent: editor.doc.selection,
                search: String::new(),
            });
        }
        ui.add(
            egui::TextEdit::singleline(&mut editor.tree.filter)
                .hint_text("Filter Nodes")
                .desired_width(f32::INFINITY),
        );
    });
    ui.add_space(2.0);

    if editor.doc.scene.nodes.is_empty() {
        empty_scene_ui(ui, editor);
        return;
    }

    let mut actions = Vec::new();
    let filter = editor.tree.filter.to_lowercase();
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        let visible = visible_rows(&editor.doc, &editor.tree.collapsed, &filter);
        for (id, depth, has_children) in visible {
            row(ui, editor, id, depth, has_children, &mut actions);
        }
        // The space below the rows: dropping there moves a node to the top level, and
        // clicking there clears the selection.
        let rest = ui.available_size().max(vec2(0.0, 40.0));
        let (rect, response) = ui.allocate_exact_size(rest, Sense::click());
        if response.clicked() {
            editor.doc.selection = None;
        }
        if let Some(dragged) = response.dnd_release_payload::<NodeDrag>() {
            actions.push(TreeAction::Move(dragged.0, DropPlace::Root));
        } else if response.dnd_hover_payload::<NodeDrag>().is_some() {
            ui.painter().hline(rect.x_range(), rect.top(), Stroke::new(2.0, theme::ACCENT));
        }
    });

    for action in actions {
        match action {
            TreeAction::Select(id) => editor.doc.selection = Some(id),
            TreeAction::ToggleCollapse(id) => {
                if !editor.tree.collapsed.remove(&id) {
                    editor.tree.collapsed.insert(id);
                }
            }
            TreeAction::ToggleVisible(id) => {
                if let Some(node) = editor.doc.node_mut(id) {
                    node.visible = !node.visible;
                }
            }
            TreeAction::Move(id, place) => {
                if !editor.doc.move_node(id, place) {
                    editor.log.error("A node can't be moved into itself.");
                }
            }
            TreeAction::Rename(id, name) => editor.doc.rename(id, &name),
        }
    }
}

fn empty_scene_ui(ui: &mut Ui, editor: &mut Editor) {
    ui.add_space(8.0);
    ui.label(RichText::new("Create Root Node:").strong());
    ui.add_space(2.0);
    for index in 0..editor::NODE_TYPES.len() {
        let label = format!("{}  {}", type_icon(index), editor::NODE_TYPES[index].0);
        if ui.add(egui::Button::new(label).min_size(vec2(ui.available_width(), 24.0))).clicked() {
            editor::add_node_of_type(&mut editor.doc, None, index);
        }
    }
    ui.add_space(8.0);
    ui.label(
        RichText::new("Or double-click a model or scene in the FileSystem dock.")
            .color(theme::TEXT_DIM),
    );
}

/// Rows to show, in order: (node, depth, has children).
fn visible_rows(doc: &Document, collapsed: &HashSet<NodeId>, filter: &str) -> Vec<(NodeId, usize, bool)> {
    let mut rows = Vec::new();
    if filter.is_empty() {
        fn walk(
            doc: &Document,
            parent: Option<NodeId>,
            depth: usize,
            collapsed: &HashSet<NodeId>,
            rows: &mut Vec<(NodeId, usize, bool)>,
        ) {
            for node in doc.children(parent) {
                let has_children = doc.children(Some(node.id)).next().is_some();
                rows.push((node.id, depth, has_children));
                if !collapsed.contains(&node.id) {
                    walk(doc, Some(node.id), depth + 1, collapsed, rows);
                }
            }
        }
        walk(doc, None, 0, collapsed, &mut rows);
    } else {
        // Filtering shows matches as a flat list.
        for node in &doc.scene.nodes {
            if node.name.to_lowercase().contains(filter) {
                rows.push((node.id, 0, false));
            }
        }
    }
    rows
}

#[derive(Clone, Copy)]
struct NodeDrag(NodeId);

fn row(
    ui: &mut Ui,
    editor: &mut Editor,
    id: NodeId,
    depth: usize,
    has_children: bool,
    actions: &mut Vec<TreeAction>,
) {
    let Some(node) = editor.doc.node(id).cloned() else {
        return;
    };
    let (rect, response) =
        ui.allocate_exact_size(vec2(ui.available_width(), ROW_HEIGHT), Sense::click_and_drag());
    let selected = editor.doc.selection == Some(id);
    let indent = 4.0 + depth as f32 * INDENT;

    let arrow_rect = Rect::from_min_size(Pos2::new(rect.left() + indent, rect.top()), vec2(14.0, ROW_HEIGHT));
    let eye_rect = Rect::from_min_size(Pos2::new(rect.right() - 24.0, rect.top()), vec2(22.0, ROW_HEIGHT));

    if selected {
        ui.painter().rect_filled(rect, 2.0, theme::SELECTED);
    } else if response.hovered() {
        ui.painter().rect_filled(rect, 2.0, theme::HOVER);
    }

    let pointer = ui.input(|i| i.pointer.interact_pos());
    let on = |r: Rect| pointer.is_some_and(|p| r.contains(p));
    let name_x = paint_row(ui, editor, &node, rect, arrow_rect, eye_rect, has_children);

    if response.clicked() {
        if has_children && on(arrow_rect) {
            actions.push(TreeAction::ToggleCollapse(id));
        } else if on(eye_rect) {
            actions.push(TreeAction::ToggleVisible(id));
        } else {
            actions.push(TreeAction::Select(id));
        }
    }
    if response.double_clicked() && !on(eye_rect) && !on(arrow_rect) {
        editor.tree.begin_rename(id, node.name.clone());
    }

    rename_field(ui, editor, id, rect, name_x, actions);

    response.dnd_set_drag_payload(NodeDrag(id));
    if let Some(hover) = response.dnd_hover_payload::<NodeDrag>() {
        if hover.0 != id {
            let place = drop_place(ui, id, rect);
            draw_drop_hint(ui, rect, place);
        }
    }
    if let Some(dragged) = response.dnd_release_payload::<NodeDrag>() {
        if dragged.0 != id {
            actions.push(TreeAction::Move(dragged.0, drop_place(ui, id, rect)));
        }
    }

    response.context_menu(|ui| {
        editor.doc.selection = Some(id);
        if ui.button("Add Child Node…").clicked() {
            editor.dialog = Some(Dialog::CreateNode { parent: Some(id), search: String::new() });
            ui.close();
        }
        if ui.button("Rename").clicked() {
            editor.tree.begin_rename(id, node.name.clone());
            ui.close();
        }
        if ui.button("Duplicate").clicked() {
            editor.actions.push(Action::DuplicateSelected);
            ui.close();
        }
        ui.separator();
        if ui.button("Delete").clicked() {
            editor.actions.push(Action::DeleteSelected);
            ui.close();
        }
    });
}

fn paint_row(
    ui: &Ui,
    editor: &Editor,
    node: &SceneNode,
    rect: Rect,
    arrow_rect: Rect,
    eye_rect: Rect,
    has_children: bool,
) -> f32 {
    let painter = ui.painter();
    let y = rect.center().y;
    let font = FontId::proportional(13.0);
    if has_children {
        let arrow = if editor.tree.collapsed.contains(&node.id) { "⏵" } else { "⏷" };
        painter.text(arrow_rect.center(), egui::Align2::CENTER_CENTER, arrow, font.clone(), theme::TEXT_DIM);
    }
    let (icon, color) = kind_icon(&node.kind);
    let icon_x = arrow_rect.right() + 2.0;
    painter.text(Pos2::new(icon_x, y), egui::Align2::LEFT_CENTER, icon, font.clone(), color);
    let name_x = icon_x + 20.0;
    let text_color = if node.visible { theme::TEXT } else { theme::TEXT_DIM };
    painter.text(Pos2::new(name_x, y), egui::Align2::LEFT_CENTER, &node.name, font.clone(), text_color);

    let (eye, eye_color) = if node.visible { ("👁", theme::TEXT_DIM) } else { ("—", theme::ACTIVE) };
    painter.text(eye_rect.center(), egui::Align2::CENTER_CENTER, eye, font, eye_color);
    name_x
}

fn rename_field(
    ui: &mut Ui,
    editor: &mut Editor,
    id: NodeId,
    rect: Rect,
    name_x: f32,
    actions: &mut Vec<TreeAction>,
) {
    let needs_focus = editor.tree.rename_needs_focus;
    let Some((rename_id, text)) = &mut editor.tree.rename else {
        return;
    };
    if *rename_id != id {
        return;
    }
    let field_rect = Rect::from_min_max(Pos2::new(name_x - 2.0, rect.top()), Pos2::new(rect.right() - 26.0, rect.bottom()));
    let field = ui.put(field_rect, egui::TextEdit::singleline(text).margin(egui::Margin::symmetric(2, 1)));
    let cancel = ui.input(|i| i.key_pressed(egui::Key::Escape));
    let finished = field.lost_focus();
    let name = text.clone();
    if needs_focus {
        field.request_focus();
        editor.tree.rename_needs_focus = false;
    } else if finished || cancel {
        if !cancel {
            actions.push(TreeAction::Rename(id, name));
        }
        editor.tree.rename = None;
    }
}

fn drop_place(ui: &Ui, id: NodeId, rect: Rect) -> DropPlace {
    let y = ui.input(|i| i.pointer.interact_pos()).map_or(rect.center().y, |p| p.y);
    let t = (y - rect.top()) / rect.height();
    if t < 0.25 {
        DropPlace::Before(id)
    } else if t > 0.75 {
        DropPlace::After(id)
    } else {
        DropPlace::ChildOf(id)
    }
}

fn draw_drop_hint(ui: &Ui, rect: Rect, place: DropPlace) {
    let stroke = Stroke::new(2.0, theme::ACCENT);
    match place {
        DropPlace::Before(_) => {
            ui.painter().hline(rect.x_range(), rect.top(), stroke);
        }
        DropPlace::After(_) => {
            ui.painter().hline(rect.x_range(), rect.bottom(), stroke);
        }
        _ => {
            ui.painter().rect_stroke(rect, 2.0, stroke, egui::StrokeKind::Inside);
        }
    }
}
