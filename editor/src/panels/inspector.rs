//! The Inspector dock: properties of the selected node, or of the scene's environment.

use egui::{Color32, DragValue, RichText, Ui};
use vulkano_engine::scene::scene_file::{
    BodyKind, ColliderShape, Environment, NodeKind, PhysicsDesc,
};

use crate::editor::Editor;
use crate::panels::scene_tree::kind_icon;
use crate::theme;

const LABEL_WIDTH: f32 = 96.0;

pub fn ui(ui: &mut Ui, editor: &mut Editor) {
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        if editor.doc.selection.is_some() {
            node_ui(ui, editor);
        } else {
            environment_ui(ui, &mut editor.doc.scene.environment);
        }
    });
}

/// A label on the left and the widget on the right, like Godot's property rows.
fn property(ui: &mut Ui, label: &str, contents: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        ui.add_sized([LABEL_WIDTH, 22.0], egui::Label::new(RichText::new(label).color(theme::TEXT_DIM)).truncate());
        contents(ui);
    });
}

fn section(ui: &mut Ui, title: &str, contents: impl FnOnce(&mut Ui)) {
    egui::CollapsingHeader::new(RichText::new(title).strong())
        .default_open(true)
        .show(ui, contents);
}

fn axis_value(ui: &mut Ui, axis: &str, color: Color32, value: &mut f32, speed: f64, suffix: &str) {
    ui.label(RichText::new(axis).color(color).strong());
    ui.add(DragValue::new(value).speed(speed).max_decimals(3).suffix(suffix));
}

fn vec3_property(ui: &mut Ui, label: &str, value: &mut [f32; 3], speed: f64, suffix: &str) {
    property(ui, label, |ui| {
        ui.spacing_mut().item_spacing.x = 3.0;
        axis_value(ui, "x", theme::AXIS_X, &mut value[0], speed, suffix);
        axis_value(ui, "y", theme::AXIS_Y, &mut value[1], speed, suffix);
        axis_value(ui, "z", theme::AXIS_Z, &mut value[2], speed, suffix);
    });
}

fn float_property(ui: &mut Ui, label: &str, value: &mut f32, speed: f64, min: f32, max: f32) {
    property(ui, label, |ui| {
        ui.add(DragValue::new(value).speed(speed).range(min..=max).max_decimals(3));
    });
}

/// Colours here are light energy and may go above 1, so they stay plain numbers,
/// with a swatch showing the normalized hue.
fn color_property(ui: &mut Ui, label: &str, value: &mut [f32; 3]) {
    property(ui, label, |ui| {
        ui.spacing_mut().item_spacing.x = 3.0;
        let peak = value.iter().copied().fold(1.0_f32, f32::max);
        let [r, g, b] = value.map(|c| (c / peak).clamp(0.0, 1.0));
        let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 3.0, egui::Rgba::from_rgb(r, g, b));
        for (axis, component) in ["r", "g", "b"].iter().zip(value.iter_mut()) {
            ui.label(RichText::new(*axis).color(theme::TEXT_DIM));
            ui.add(DragValue::new(component).speed(0.02).range(0.0..=1000.0).max_decimals(2));
        }
    });
}

fn node_ui(ui: &mut Ui, editor: &mut Editor) {
    let models = &editor.models;
    let Some(id) = editor.doc.selection else {
        return;
    };
    let parent_visible = editor.doc.scene.effective_visibility();
    let Some(node) = editor.doc.node_mut(id) else {
        return;
    };

    let (icon, color) = kind_icon(&node.kind);
    let type_name = match node.kind {
        NodeKind::Empty => "Node3D",
        NodeKind::Model { .. } => "Model",
        NodeKind::PointLight { .. } => "PointLight",
    };
    ui.horizontal(|ui| {
        ui.label(RichText::new(icon).color(color));
        ui.label(RichText::new(&node.name).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(type_name).color(theme::TEXT_DIM));
        });
    });
    ui.separator();

    match &mut node.kind {
        NodeKind::Model { path, animation } => section(ui, "Model", |ui| {
            property(ui, "File", |ui| {
                ui.add(egui::Label::new(RichText::new(path.as_str()).monospace()).truncate());
            });
            match models.get(path.as_str()) {
                Some(info) => {
                    property(ui, "Triangles", |ui| {
                        ui.label(info.triangles.to_string());
                    });
                    if !info.clips.is_empty() {
                        property(ui, "Animation", |ui| {
                            let current = animation.clone().unwrap_or_else(|| "[none]".to_string());
                            egui::ComboBox::from_id_salt("animation")
                                .selected_text(current)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(animation, None, "[none]");
                                    for clip in &info.clips {
                                        ui.selectable_value(animation, Some(clip.clone()), clip);
                                    }
                                });
                        });
                    }
                }
                None => {
                    ui.label(RichText::new("This model has not loaded.").color(theme::WARNING));
                }
            }
        }),
        NodeKind::PointLight { color, intensity, radius } => section(ui, "PointLight", |ui| {
            color_property(ui, "Color", color);
            float_property(ui, "Intensity", intensity, 0.05, 0.0, 10000.0);
            float_property(ui, "Radius", radius, 0.05, 0.01, 10000.0);
        }),
        NodeKind::Empty => {
            if node.name != "PlayerSpawn" {
                ui.label(
                    RichText::new("Tip: name a Node3D PlayerSpawn to set where the game starts.")
                        .color(theme::TEXT_DIM),
                );
            } else {
                ui.label(RichText::new("The game's player starts here.").color(theme::PLAY));
            }
        }
    }

    section(ui, "Transform", |ui| {
        vec3_property(ui, "Position", &mut node.transform.position, 0.02, "");
        vec3_property(ui, "Rotation", &mut node.transform.rotation, 0.5, "°");
        float_property(ui, "Scale", &mut node.transform.scale, 0.01, 0.0001, 100000.0);
    });

    section(ui, "Visibility", |ui| {
        property(ui, "Visible", |ui| {
            ui.checkbox(&mut node.visible, "");
        });
        if node.visible && !parent_visible.get(&id).copied().unwrap_or(true) {
            ui.label(RichText::new("Hidden because a parent is hidden.").color(theme::WARNING));
        }
    });

    section(ui, "Physics", |ui| {
        let mut enabled = node.physics.is_some();
        property(ui, "Enabled", |ui| {
            ui.checkbox(&mut enabled, "");
        });
        if enabled && node.physics.is_none() {
            node.physics = Some(PhysicsDesc {
                body: BodyKind::Fixed,
                collider: ColliderShape::Cuboid { half_extents: [0.5, 0.5, 0.5] },
            });
        } else if !enabled {
            node.physics = None;
        }
        if let Some(physics) = &mut node.physics {
            property(ui, "Body", |ui| {
                egui::ComboBox::from_id_salt("body")
                    .selected_text(format!("{:?}", physics.body))
                    .show_ui(ui, |ui| {
                        for kind in [BodyKind::Fixed, BodyKind::Dynamic, BodyKind::Kinematic] {
                            ui.selectable_value(&mut physics.body, kind, format!("{kind:?}"));
                        }
                    });
            });
            collider_ui(ui, &mut physics.collider);
            ui.label(
                RichText::new("Physics runs when you press Play. Colliders are drawn in the 3D view when selected.")
                    .color(theme::TEXT_DIM)
                    .small(),
            );
        }
    });
}

fn collider_ui(ui: &mut Ui, collider: &mut ColliderShape) {
    let name = match collider {
        ColliderShape::Cuboid { .. } => "Box",
        ColliderShape::Ball { .. } => "Sphere",
        ColliderShape::Capsule { .. } => "Capsule",
    };
    property(ui, "Shape", |ui| {
        egui::ComboBox::from_id_salt("collider").selected_text(name).show_ui(ui, |ui| {
            if ui.selectable_label(name == "Box", "Box").clicked() {
                *collider = ColliderShape::Cuboid { half_extents: [0.5, 0.5, 0.5] };
            }
            if ui.selectable_label(name == "Sphere", "Sphere").clicked() {
                *collider = ColliderShape::Ball { radius: 0.5 };
            }
            if ui.selectable_label(name == "Capsule", "Capsule").clicked() {
                *collider = ColliderShape::Capsule { half_height: 0.5, radius: 0.5 };
            }
        });
    });
    match collider {
        ColliderShape::Cuboid { half_extents } => {
            vec3_property(ui, "Half Extents", half_extents, 0.02, "");
        }
        ColliderShape::Ball { radius } => float_property(ui, "Radius", radius, 0.01, 0.001, 10000.0),
        ColliderShape::Capsule { half_height, radius } => {
            float_property(ui, "Half Height", half_height, 0.01, 0.001, 10000.0);
            float_property(ui, "Radius", radius, 0.01, 0.001, 10000.0);
        }
    }
}

fn environment_ui(ui: &mut Ui, env: &mut Environment) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("🌐").color(theme::ACCENT));
        ui.label(RichText::new("Environment").strong());
    });
    ui.label(RichText::new("Nothing selected. These settings apply to the whole scene.").color(theme::TEXT_DIM).small());
    ui.separator();

    section(ui, "Sun", |ui| {
        vec3_property(ui, "Direction", &mut env.sun_direction, 0.01, "");
        color_property(ui, "Color", &mut env.sun_color);
        float_property(ui, "Shadow Softness", &mut env.shadow_softness, 0.05, 0.0, 16.0);
    });
    section(ui, "Ambient", |ui| {
        color_property(ui, "Color", &mut env.ambient_color);
        float_property(ui, "Intensity", &mut env.ambient_intensity, 0.01, 0.0, 100.0);
    });
    section(ui, "Camera Effects", |ui| {
        float_property(ui, "Exposure", &mut env.exposure, 0.01, 0.0, 100.0);
        property(ui, "FXAA", |ui| {
            ui.checkbox(&mut env.fxaa, "");
        });
        float_property(ui, "AO Radius", &mut env.ao_radius, 0.002, 0.0, 5.0);
        float_property(ui, "AO Strength", &mut env.ao_strength, 0.01, 0.0, 1.0);
    });
}
