//! The Inspector dock: an entity's components, or the scene's environment when nothing
//! is selected. Fields are read from and written straight to the components in the world.

use egui::{Button, DragValue, RichText, Ui};
use shipyard::{EntityId, World};
use vulkano_engine::assets::asset_manager::AssetManager;
use vulkano_engine::scene::components::object3d::Object3D;
use vulkano_engine::scene::scene_file::{
    ComponentDef, Environment, Field, FieldHint, FieldValue,
};
use vulkano_engine::scripting::ScriptInstance;

use crate::editor::{Dialog, Editor};
use crate::panels::entities::entity_icon;
use crate::panels::filesystem::{self, ModelDrag};
use crate::panels::{Picked, search_combo};
use crate::theme;

const LABEL_WIDTH: f32 = 96.0;

/// Euler angles aren't unique, so what's read back from a rotation matrix may differ from
/// what was typed (170° past 90° on X flips to other values). While the rotation hasn't
/// changed from elsewhere, keep showing the numbers the user is dragging.
#[derive(Default)]
pub struct InspectorState {
    euler: Option<EulerEdit>,
    /// Components added in this session, in order, per entity. They're listed last.
    pub added: std::collections::HashMap<EntityId, Vec<&'static str>>,
}

struct EulerEdit {
    entity: EntityId,
    shown: [f32; 3],
    stored: [f32; 3],
}

pub fn ui(ui: &mut Ui, editor: &mut Editor, world: &mut World) {
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        match editor.doc.selection {
            Some(entity) => entity_ui(ui, editor, world, entity),
            None => {
                environment_ui(ui, &mut editor.doc.environment);
                post_effects_ui(ui, editor);
            }
        }
    });
}

/// A label on the left and the widget on the right, like Godot's property rows.
fn property(ui: &mut Ui, label: &str, contents: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [LABEL_WIDTH, 22.0],
            egui::Label::new(RichText::new(label).color(theme::text_dim())).truncate(),
        );
        contents(ui);
    });
}

fn pretty(name: &str) -> String {
    let mut out = String::new();
    for (i, word) in name.split('_').enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.push_str(chars.as_str());
        }
    }
    out
}

fn vec3_widgets(ui: &mut Ui, value: &mut [f32; 3], labels: [(&str, egui::Color32); 3], suffix: &str, speed: f64) -> bool {
    let mut changed = false;
    ui.spacing_mut().item_spacing.x = 3.0;
    for ((label, color), component) in labels.into_iter().zip(value.iter_mut()) {
        ui.label(RichText::new(label).color(color).strong());
        changed |= ui
            .add(DragValue::new(component).speed(speed).max_decimals(3).suffix(suffix))
            .changed();
    }
    changed
}

fn xyz() -> [(&'static str, egui::Color32); 3] {
    [("x", theme::axis_x()), ("y", theme::axis_y()), ("z", theme::axis_z())]
}

fn rgb_labels() -> [(&'static str, egui::Color32); 3] {
    [("r", theme::text_dim()), ("g", theme::text_dim()), ("b", theme::text_dim())]
}

/// Draw a widget for a value; returns the new value if it changed.
fn value_ui(
    ui: &mut Ui,
    value: &FieldValue,
    hint: FieldHint,
    clips: &[String],
    assets: &[String],
) -> Option<FieldValue> {
    match (value, hint) {
        (FieldValue::Text(current), FieldHint::Choice(options)) => {
            let mut picked = None;
            egui::ComboBox::from_id_salt(ui.next_auto_id()).selected_text(current.as_str()).show_ui(ui, |ui| {
                for option in options {
                    if ui.selectable_label(current == option, *option).clicked() {
                        picked = Some(FieldValue::Text(option.to_string()));
                    }
                }
            });
            picked
        }
        (FieldValue::Text(current), FieldHint::AnimationClip) => {
            let shown = if current.is_empty() { "[none]" } else { current.as_str() };
            let mut labels = vec!["[none]".to_string()];
            labels.extend(clips.iter().cloned());
            let at = labels.iter().position(|l| l == current).or(current.is_empty().then_some(0));
            match search_combo(ui, ui.next_auto_id(), shown, 160.0, &labels, at, None) {
                Some(Picked::Item(0)) => Some(FieldValue::Text(String::new())),
                Some(Picked::Item(i)) => Some(FieldValue::Text(labels[i].clone())),
                _ => None,
            }
        }
        (FieldValue::Text(current), FieldHint::Asset(_)) => {
            let shown = if current.is_empty() {
                "[choose a file]".to_string()
            } else {
                filesystem::file_name(current)
            };
            let labels: Vec<String> = assets.iter().map(|a| filesystem::file_name(a)).collect();
            let at = assets.iter().position(|a| a == current);
            let width = ui.available_width().min(220.0);
            match search_combo(ui, ui.next_auto_id(), &shown, width, &labels, at, None) {
                Some(Picked::Item(i)) => Some(FieldValue::Text(assets[i].clone())),
                _ => None,
            }
        }
        (FieldValue::Text(current), _) => {
            let mut text = current.clone();
            ui.add(egui::TextEdit::singleline(&mut text).desired_width(180.0))
                .changed()
                .then_some(FieldValue::Text(text))
        }
        (FieldValue::Float(current), hint) => {
            let mut v = *current;
            let mut drag = DragValue::new(&mut v).speed(0.02).max_decimals(3);
            if let FieldHint::Range(min, max) = hint {
                drag = drag.range(min..=max);
            }
            ui.add(drag).changed().then_some(FieldValue::Float(v))
        }
        (FieldValue::Int(current), FieldHint::Layers) => {
            let names = layer_names();
            let mut bits = *current as u32;
            let mut changed = false;
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
                for row in 0..2 {
                    ui.horizontal(|ui| {
                        for column in 0..8 {
                            let layer = row * 8 + column;
                            let on = bits & (1 << layer) != 0;
                            let button = egui::Button::selectable(on, format!("{}", layer + 1)).min_size(egui::vec2(22.0, 18.0));
                            let tip = names.get(layer).filter(|n| !n.is_empty()).cloned().unwrap_or_else(|| format!("Layer {}", layer + 1));
                            if ui.add(button).on_hover_text(tip).clicked() {
                                bits ^= 1 << layer;
                                changed = true;
                            }
                        }
                    });
                }
            });
            changed.then_some(FieldValue::Int(bits as i64))
        }
        (FieldValue::Int(current), _) => {
            let mut v = *current;
            ui.add(DragValue::new(&mut v)).changed().then_some(FieldValue::Int(v))
        }
        (FieldValue::Bool(current), _) => {
            let mut v = *current;
            ui.checkbox(&mut v, "").changed().then_some(FieldValue::Bool(v))
        }
        (FieldValue::Vec3(current), FieldHint::Color) => {
            let mut v = *current;
            // Light energy may go above 1, so show the hue normalized beside the numbers.
            let peak = v.iter().copied().fold(1.0_f32, f32::max);
            let [r, g, b] = v.map(|c| (c / peak).clamp(0.0, 1.0));
            let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 3.0, egui::Rgba::from_rgb(r, g, b));
            vec3_widgets(ui, &mut v, rgb_labels(), "", 0.02).then_some(FieldValue::Vec3(v))
        }
        (FieldValue::Vec3(current), hint) => {
            let mut v = *current;
            let (suffix, speed) = if hint == FieldHint::Degrees { ("°", 0.5) } else { ("", 0.02) };
            vec3_widgets(ui, &mut v, xyz(), suffix, speed).then_some(FieldValue::Vec3(v))
        }
        (FieldValue::Scripts(_), _) => None,
    }
}

fn entity_ui(ui: &mut Ui, editor: &mut Editor, world: &mut World, entity: EntityId) {
    let (icon, color) = entity_icon(world, entity);
    let name = editor.doc.display_name(world, entity);
    ui.horizontal(|ui| {
        ui.label(RichText::new(icon).color(color));
        ui.label(RichText::new(&name).strong());
    });
    ui.separator();

    let defs: Vec<ComponentDef> = editor.registry.defs().to_vec();
    let mut remove = None;
    let mut model_changed = false;

    let mut present: Vec<ComponentDef> = defs.iter().filter(|d| d.has(world, entity)).copied().collect();
    let added = editor.inspector.added.get(&entity).cloned().unwrap_or_default();
    present.sort_by_key(|d| added.iter().position(|n| *n == d.name).map_or(0, |i| i + 1));
    for def in present.iter() {
        let id = ui.make_persistent_id(("component", def.name));
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true)
            .show_header(ui, |ui| {
                ui.label(RichText::new(def.name).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("✖").on_hover_text(format!("Remove {}", def.name)).clicked() {
                        remove = Some(*def);
                    }
                });
            })
            .body(|ui| {
                let fields = def.fields(world, entity).unwrap_or_default();
                if fields.is_empty() {
                    ui.label(RichText::new("No editable values.").color(theme::text_dim()).small());
                }
                for field in fields {
                    if matches!(field.value, FieldValue::Scripts(_)) {
                        scripts_ui(ui, editor, world, entity, def, &field);
                    } else if let Some(value) = field_row(ui, editor, world, entity, &field) {
                        def.set_field(world, entity, field.name, &value);
                        remember_euler(editor, world, entity, def, &field, &value);
                        if def.name == "Object3D" && field.name == "model" {
                            model_changed = true;
                        }
                    }
                }
                if def.name == "Transform" {
                    ui.label(RichText::new("Engine space: -Y is up.").color(theme::text_dim()).small());
                }
            });
        ui.add_space(2.0);
    }

    if model_changed {
        // The old animator's skeleton belongs to the old model.
        editor.registry.def("Animator").unwrap().remove(world, entity);
        editor.doc.wants_animator.insert(entity);
    }
    if let Some(def) = remove {
        def.remove(world, entity);
    }

    ui.separator();
    add_component_ui(ui, editor, world, entity, &defs);

    if let Ok(object) = world.get::<&Object3D>(entity) {
        if let Some(info) = editor.models.get(&object.model.id) {
            ui.add_space(4.0);
            ui.label(RichText::new(format!("{} triangles", info.triangles)).color(theme::text_dim()).small());
        }
    }
}

fn field_row(
    ui: &mut Ui,
    editor: &mut Editor,
    world: &World,
    entity: EntityId,
    field: &Field,
) -> Option<FieldValue> {
    let clips: Vec<String> = match field.hint {
        FieldHint::AnimationClip => world
            .get::<&Object3D>(entity)
            .ok()
            .and_then(|o| editor.models.get(&o.model.id).map(|m| m.clips.clone()))
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    let assets: Vec<String> = match field.hint {
        // Shader fields list material shaders, not post effects.
        FieldHint::Asset(extensions) => editor
            .files
            .files_with_extensions(extensions)
            .into_iter()
            .filter(|path| !extensions.contains(&"glsl") || !path.contains("/post/"))
            .collect(),
        _ => Vec::new(),
    };

    // Show the numbers being dragged while the rotation hasn't changed from elsewhere.
    let mut value = field.value.clone();
    if let (FieldHint::Degrees, FieldValue::Vec3(stored), Some(edit)) = (field.hint, &field.value, &editor.inspector.euler) {
        if edit.entity == entity && edit.stored == *stored {
            value = FieldValue::Vec3(edit.shown);
        }
    }

    let mut result = None;
    let row = ui.horizontal(|ui| {
        ui.add_sized(
            [LABEL_WIDTH, 22.0],
            egui::Label::new(RichText::new(pretty(field.name)).color(theme::text_dim())).truncate(),
        );
        result = value_ui(ui, &value, field.hint, &clips, &assets);
    });

    if let (FieldHint::Asset(_), FieldValue::Text(path)) = (field.hint, &field.value) {
        if let Some((_, error)) = editor.shader_errors.iter().find(|(p, _)| p == path) {
            ui.label(RichText::new(error).color(theme::error()).monospace().small());
        }
    }

    // Models can also be dragged here from the FileSystem dock.
    if matches!(field.hint, FieldHint::Asset(_)) {
        if let Some(payload) = row.response.dnd_release_payload::<ModelDrag>() {
            result = Some(FieldValue::Text(payload.0.clone()));
        }
    }
    result
}

fn remember_euler(
    editor: &mut Editor,
    world: &World,
    entity: EntityId,
    def: &ComponentDef,
    field: &Field,
    value: &FieldValue,
) {
    if field.hint != FieldHint::Degrees {
        return;
    }
    let stored = def
        .fields(world, entity)
        .and_then(|fields| fields.into_iter().find(|f| f.name == field.name))
        .and_then(|f| f.value.as_vec3());
    if let (Some(stored), Some(shown)) = (stored, value.as_vec3()) {
        editor.inspector.euler = Some(EulerEdit { entity, shown, stored });
    }
}

fn add_component_ui(ui: &mut Ui, editor: &mut Editor, world: &mut World, entity: EntityId, defs: &[ComponentDef]) {
    let available: Vec<ComponentDef> = defs.iter().filter(|d| !d.has(world, entity)).copied().collect();
    let labels: Vec<String> = available.iter().map(|d| d.name.to_string()).collect();
    let width = ui.available_width().min(240.0);
    if let Some(Picked::Item(i)) = search_combo(ui, "add_component", "✚ Add Component", width, &labels, None, None) {
        let def = available[i];
        match def.add(world, entity) {
            // New components go to the bottom of the list, where the user is looking.
            Ok(()) => editor.inspector.added.entry(entity).or_default().push(def.name),
            Err(e) => editor.log.error(e),
        }
    }
}

fn scripts_ui(
    ui: &mut Ui,
    editor: &mut Editor,
    world: &World,
    entity: EntityId,
    def: &ComponentDef,
    field: &Field,
) {
    let FieldValue::Scripts(current) = &field.value else {
        return;
    };
    let mut instances = current.clone();
    let mut changed = false;
    let mut remove = None;

    for (index, instance) in instances.iter_mut().enumerate() {
        ui.push_id(index, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(filesystem::file_name(&instance.path)).strong().color(theme::accent()));
                if ui.small_button("Open").on_hover_text("Open in your code editor").clicked() {
                    editor.scripts.open_in_editor(&instance.path, &mut editor.log);
                }
                if ui.small_button("✖").on_hover_text("Detach this script").clicked() {
                    remove = Some(index);
                }
            });
            changed |= exports_ui(ui, editor, instance);
        });
        ui.add_space(2.0);
    }
    if let Some(index) = remove {
        instances.remove(index);
        changed = true;
    }

    // Attach an existing script, or start a new one.
    let mut attach = None;
    let files = editor.scripts.files().to_vec();
    let labels: Vec<String> = files.iter().map(|f| filesystem::file_name(f)).collect();
    let width = ui.available_width().min(240.0);
    match search_combo(ui, "attach_script", "✚ Attach Script", width, &labels, None, Some("New Script…")) {
        Some(Picked::Item(i)) => attach = Some(files[i].clone()),
        Some(Picked::Extra) => {
            editor.dialog = Some(Dialog::NewFile { kind: crate::panels::scripts::FileKind::Script, name: String::new() });
        }
        None => {}
    }
    if let Some(path) = attach {
        instances.push(ScriptInstance { path, ..Default::default() });
        changed = true;
    }

    if changed {
        def.set_field(world, entity, field.name, &FieldValue::Scripts(instances));
    }
}

/// The values a script exports. Only values changed from the script's default are stored
/// on the entity, so editing a default in the script reaches every untouched entity.
fn exports_ui(ui: &mut Ui, editor: &mut Editor, instance: &mut ScriptInstance) -> bool {
    let exports = match editor.scripts.exports(&instance.path) {
        Ok(exports) => exports.clone(),
        Err(problem) => {
            ui.label(RichText::new(problem.as_str()).color(theme::error()).small());
            return false;
        }
    };
    if exports.is_empty() {
        ui.label(RichText::new("This script has no exported values.").color(theme::text_dim()).small());
    }
    let mut changed = false;
    for export in exports {
        let is_overridden = instance.exports.contains_key(&export.name);
        let value = instance.exports.get(&export.name).unwrap_or(&export.default).clone();
        let mut picked = None;
        let mut reset = false;
        property(ui, &pretty(&export.name), |ui| {
            picked = value_ui(ui, &value, FieldHint::None, &[], &[]);
            if is_overridden && ui.add(Button::new("⟲").small()).on_hover_text("Back to the script's default").clicked() {
                reset = true;
            }
        });
        if let Some(new) = picked {
            if new == export.default {
                instance.exports.remove(&export.name);
            } else {
                instance.exports.insert(export.name.clone(), new);
            }
            changed = true;
        }
        if reset {
            instance.exports.remove(&export.name);
            changed = true;
        }
    }
    changed
}

fn environment_ui(ui: &mut Ui, env: &mut Environment) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("🌐").color(theme::accent()));
        ui.label(RichText::new("Environment").strong());
    });
    ui.label(RichText::new("Nothing selected. These settings apply to the whole scene.").color(theme::text_dim()).small());
    ui.separator();

    let color = |ui: &mut Ui, label: &str, value: &mut [f32; 3]| {
        property(ui, label, |ui| {
            if let Some(FieldValue::Vec3(new)) = value_ui(ui, &FieldValue::Vec3(*value), FieldHint::Color, &[], &[]) {
                *value = new;
            }
        });
    };
    let float = |ui: &mut Ui, label: &str, value: &mut f32, speed: f64, min: f32, max: f32| {
        property(ui, label, |ui| {
            ui.add(DragValue::new(value).speed(speed).range(min..=max).max_decimals(3));
        });
    };

    egui::CollapsingHeader::new(RichText::new("Sun").strong()).default_open(true).show(ui, |ui| {
        property(ui, "Direction", |ui| {
            vec3_widgets(ui, &mut env.sun_direction, xyz(), "", 0.01);
        });
        color(ui, "Color", &mut env.sun_color);
        float(ui, "Shadow Softness", &mut env.shadow_softness, 0.05, 0.0, 16.0);
    });
    egui::CollapsingHeader::new(RichText::new("Ambient").strong()).default_open(true).show(ui, |ui| {
        color(ui, "Color", &mut env.ambient_color);
        float(ui, "Intensity", &mut env.ambient_intensity, 0.01, 0.0, 100.0);
    });
    egui::CollapsingHeader::new(RichText::new("Camera Effects").strong()).default_open(true).show(ui, |ui| {
        float(ui, "Exposure", &mut env.exposure, 0.01, 0.0, 100.0);
        property(ui, "FXAA", |ui| {
            ui.checkbox(&mut env.fxaa, "");
        });
        float(ui, "AO Radius", &mut env.ao_radius, 0.002, 0.0, 5.0);
        float(ui, "AO Strength", &mut env.ao_strength, 0.01, 0.0, 1.0);
    });
}

/// Optional names for the 16 collision layers: line N of layers.txt names layer N.
fn layer_names() -> Vec<String> {
    std::fs::read_to_string("layers.txt")
        .map(|text| text.lines().map(|l| l.trim().to_string()).collect())
        .unwrap_or_default()
}

fn post_effects_ui(ui: &mut Ui, editor: &mut Editor) {
    egui::CollapsingHeader::new(RichText::new("Post Effects").strong()).default_open(true).show(ui, |ui| {
        let mut remove = None;
        let mut swap = None;
        let count = editor.doc.environment.post_effects.len();
        for index in 0..count {
            let path = editor.doc.environment.post_effects[index].path.clone();
            let params = editor.scripts.post_params(&path);
            let error = editor.shader_errors.iter().find(|(p, _)| *p == path).map(|(_, e)| e.clone());
            let effect = &mut editor.doc.environment.post_effects[index];
            ui.push_id(index, |ui| {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut effect.enabled, "");
                    ui.label(RichText::new(filesystem::file_name(&path)).strong().color(theme::accent()));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("✖").on_hover_text("Remove").clicked() {
                            remove = Some(index);
                        }
                        if index + 1 < count && ui.small_button("⏷").on_hover_text("Run later").clicked() {
                            swap = Some(index);
                        }
                        if index > 0 && ui.small_button("⏶").on_hover_text("Run earlier").clicked() {
                            swap = Some(index - 1);
                        }
                    });
                });
                if let Some(error) = error {
                    ui.label(RichText::new(error).color(theme::error()).monospace().small());
                }
                for param in params {
                    let value = effect.params.get(&param.name).copied().unwrap_or(param.default);
                    let mut edited = value;
                    property(ui, &pretty(&param.name), |ui| {
                        ui.add(egui::Slider::new(&mut edited, param.min..=param.max));
                    });
                    if edited != value {
                        effect.params.insert(param.name.clone(), edited);
                    }
                }
            });
            ui.add_space(2.0);
        }
        if let Some(index) = remove {
            editor.doc.environment.post_effects.remove(index);
        }
        if let Some(index) = swap {
            editor.doc.environment.post_effects.swap(index, index + 1);
        }
        let available = editor.scripts.files_of(crate::panels::scripts::FileKind::PostEffect);
        let mut added = None;
        let labels: Vec<String> = available.iter().map(|f| filesystem::file_name(f)).collect();
        let width = ui.available_width().min(240.0);
        match search_combo(ui, "add_post_effect", "✚ Add Post Effect", width, &labels, None, Some("New Post Effect…")) {
            Some(Picked::Item(i)) => added = Some(available[i].clone()),
            Some(Picked::Extra) => {
                editor.dialog = Some(Dialog::NewFile { kind: crate::panels::scripts::FileKind::PostEffect, name: String::new() });
            }
            None => {}
        }
        if let Some(path) = added {
            editor.doc.environment.post_effects.push(vulkano_engine::scene::scene_file::PostEffect {
                path,
                enabled: true,
                params: Default::default(),
            });
        }
    });
}

/// Give entities whose model just changed an `Animator`, if the new model has a skeleton.
pub fn attach_pending_animators(editor: &mut Editor, world: &mut World) {
    let pending: Vec<EntityId> = editor.doc.wants_animator.iter().copied().collect();
    for entity in pending {
        let Ok(path) = world.get::<&Object3D>(entity).map(|o| o.model.id.clone()) else {
            editor.doc.wants_animator.remove(&entity);
            continue;
        };
        if path.is_empty() {
            editor.doc.wants_animator.remove(&entity);
            continue;
        }
        if !editor.models.contains_key(&path) {
            continue; // Still loading, or it failed and will stay without an animator.
        }
        editor.doc.wants_animator.remove(&entity);
        let has_skin = world
            .get_unique::<&AssetManager>()
            .ok()
            .and_then(|a| a.get_model(&vulkano_engine::assets::asset_manager::AssetHandle { id: path }).map(|m| !m.skins.is_empty()))
            .unwrap_or(false);
        if has_skin {
            let _ = editor.registry.def("Animator").unwrap().add(world, entity);
        }
    }
}
