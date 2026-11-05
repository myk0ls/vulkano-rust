//! The Entities dock: every entity in the scene, as a flat list.

use egui::{Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Ui, vec2};
use shipyard::{EntityId, World};
use vulkano_engine::physics::physics_engine::ColliderComponent;
use vulkano_engine::scene::components::camera::Camera;
use vulkano_engine::scene::components::object3d::Object3D;
use vulkano_engine::scene::components::pointlight::Pointlight;
use vulkano_engine::scene::scene_file::Hidden;

use crate::editor::{Action, Dialog, Editor};
use crate::theme;

const ROW_HEIGHT: f32 = 22.0;

#[derive(Default)]
pub struct EntityListState {
    rename: Option<(EntityId, String)>,
    rename_needs_focus: bool,
    filter: String,
}

impl EntityListState {
    pub fn begin_rename(&mut self, entity: EntityId, name: String) {
        self.rename = Some((entity, name));
        self.rename_needs_focus = true;
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// An icon showing what kind of thing an entity is, from its components.
pub fn entity_icon(world: &World, entity: EntityId) -> (&'static str, Color32) {
    if world.get::<&Object3D>(entity).is_ok() {
        ("▣", theme::entity())
    } else if world.get::<&Pointlight>(entity).is_ok() {
        ("💡", theme::warning())
    } else if world.get::<&Camera>(entity).is_ok() {
        ("🎞", theme::accent())
    } else if world.get::<&ColliderComponent>(entity).is_ok() {
        ("⛶", theme::collider())
    } else {
        ("○", theme::text_dim())
    }
}

enum ListAction {
    Select(EntityId),
    ToggleHidden(EntityId),
    Reorder(EntityId, Option<EntityId>),
    Rename(EntityId, String),
}

#[derive(Clone, Copy)]
struct EntityDrag(EntityId);

pub fn ui(ui: &mut Ui, editor: &mut Editor, world: &mut World) {
    ui.horizontal(|ui| {
        if ui.button("✚").on_hover_text("Add an entity (Ctrl+A)").clicked() {
            editor.dialog = Some(Dialog::CreateEntity { search: String::new() });
        }
        ui.add(
            egui::TextEdit::singleline(&mut editor.list.filter)
                .hint_text("Filter Entities")
                .desired_width(f32::INFINITY),
        );
    });
    ui.add_space(2.0);

    if editor.doc.order.is_empty() {
        ui.add_space(8.0);
        ui.label(RichText::new("This scene has no entities yet.").color(theme::text_dim()));
        ui.add_space(4.0);
        if ui.add(egui::Button::new("✚  Add Entity").min_size(vec2(ui.available_width(), 24.0))).clicked() {
            editor.dialog = Some(Dialog::CreateEntity { search: String::new() });
        }
        ui.add_space(4.0);
        ui.label(RichText::new("Or double-click a model or scene in the FileSystem dock.").color(theme::text_dim()));
        return;
    }

    let filter = editor.list.filter.to_lowercase();
    let mut actions = Vec::new();
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        for entity in editor.doc.order.clone() {
            let name = editor.doc.display_name(world, entity);
            if filter.is_empty() || name.to_lowercase().contains(&filter) {
                row(ui, editor, world, entity, &name, &mut actions);
            }
        }
        // Below the list: clicking clears the selection; dropping moves to the end.
        let rest = ui.available_size().max(vec2(0.0, 40.0));
        let (rect, response) = ui.allocate_exact_size(rest, Sense::click());
        if response.clicked() {
            editor.doc.selection = None;
        }
        if let Some(dragged) = response.dnd_release_payload::<EntityDrag>() {
            actions.push(ListAction::Reorder(dragged.0, None));
        } else if response.dnd_hover_payload::<EntityDrag>().is_some() {
            ui.painter().hline(rect.x_range(), rect.top(), Stroke::new(2.0, theme::accent()));
        }
    });

    for action in actions {
        match action {
            ListAction::Select(entity) => editor.doc.selection = Some(entity),
            ListAction::ToggleHidden(entity) => {
                let def = editor.registry.def("Hidden").unwrap();
                if def.has(world, entity) {
                    def.remove(world, entity);
                } else {
                    let _ = def.add(world, entity);
                }
            }
            ListAction::Reorder(entity, target) => editor.doc.reorder(entity, target),
            ListAction::Rename(entity, name) => {
                editor.doc.rename(world, &editor.registry, entity, &name)
            }
        }
    }
}

fn row(
    ui: &mut Ui,
    editor: &mut Editor,
    world: &World,
    entity: EntityId,
    name: &str,
    actions: &mut Vec<ListAction>,
) {
    let (rect, response) =
        ui.allocate_exact_size(vec2(ui.available_width(), ROW_HEIGHT), Sense::click_and_drag());
    let eye_rect = Rect::from_min_size(Pos2::new(rect.right() - 24.0, rect.top()), vec2(22.0, ROW_HEIGHT));
    let hidden = world.get::<&Hidden>(entity).is_ok();

    if editor.doc.selection == Some(entity) {
        ui.painter().rect_filled(rect, 2.0, theme::selected());
    } else if response.hovered() {
        ui.painter().rect_filled(rect, 2.0, theme::hover());
    }

    let font = FontId::proportional(13.0);
    let y = rect.center().y;
    let (icon, color) = entity_icon(world, entity);
    ui.painter().text(Pos2::new(rect.left() + 6.0, y), egui::Align2::LEFT_CENTER, icon, font.clone(), color);
    let name_x = rect.left() + 26.0;
    let text_color = if hidden { theme::text_dim() } else { theme::text() };
    ui.painter().text(Pos2::new(name_x, y), egui::Align2::LEFT_CENTER, name, font.clone(), text_color);
    let (eye, eye_color) = if hidden { ("—", theme::active()) } else { ("👁", theme::text_dim()) };
    ui.painter().text(eye_rect.center(), egui::Align2::CENTER_CENTER, eye, font, eye_color);

    let on_eye = ui.input(|i| i.pointer.interact_pos()).is_some_and(|p| eye_rect.contains(p));
    if response.clicked() {
        actions.push(if on_eye { ListAction::ToggleHidden(entity) } else { ListAction::Select(entity) });
    }
    if response.double_clicked() && !on_eye {
        editor.list.begin_rename(entity, name.to_string());
    }

    rename_field(ui, editor, entity, rect, name_x, actions);

    response.dnd_set_drag_payload(EntityDrag(entity));
    if let Some(hover) = response.dnd_hover_payload::<EntityDrag>() {
        if hover.0 != entity {
            ui.painter().hline(rect.x_range(), rect.top(), Stroke::new(2.0, theme::accent()));
        }
    }
    if let Some(dragged) = response.dnd_release_payload::<EntityDrag>() {
        actions.push(ListAction::Reorder(dragged.0, Some(entity)));
    }

    response.context_menu(|ui| {
        editor.doc.selection = Some(entity);
        if ui.button("Rename").clicked() {
            editor.list.begin_rename(entity, name.to_string());
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

fn rename_field(
    ui: &mut Ui,
    editor: &mut Editor,
    entity: EntityId,
    rect: Rect,
    name_x: f32,
    actions: &mut Vec<ListAction>,
) {
    let needs_focus = editor.list.rename_needs_focus;
    let Some((renaming, text)) = &mut editor.list.rename else {
        return;
    };
    if *renaming != entity {
        return;
    }
    let field_rect = Rect::from_min_max(Pos2::new(name_x - 2.0, rect.top()), Pos2::new(rect.right() - 26.0, rect.bottom()));
    let field = ui.put(field_rect, egui::TextEdit::singleline(text).margin(egui::Margin::symmetric(2, 1)));
    let cancel = ui.input(|i| i.key_pressed(egui::Key::Escape));
    let finished = field.lost_focus();
    let name = text.clone();
    if needs_focus {
        field.request_focus();
        editor.list.rename_needs_focus = false;
    } else if finished || cancel {
        if !cancel {
            actions.push(ListAction::Rename(entity, name));
        }
        editor.list.rename = None;
    }
}
