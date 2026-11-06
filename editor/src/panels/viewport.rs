//! The 3D dock: the rendered scene, camera controls, selection, and gizmos.

use egui::{Button, Color32, FontId, Key, PointerButton, Pos2, Rect, Sense, Stroke, Ui, vec2};
use nalgebra_glm as glm;
use transform_gizmo_egui::math::Transform as GizmoTransform;
use transform_gizmo_egui::{Gizmo, GizmoConfig, GizmoExt, GizmoMode, GizmoOrientation};
use shipyard::{EntityId, World};
use vulkano_engine::physics::physics_engine::ColliderComponent;
use vulkano_engine::scene::components::object3d::Object3D;
use vulkano_engine::scene::components::pointlight::Pointlight;
use vulkano_engine::scene::components::transform::Transform;
use vulkano_engine::scene::scene_file::Hidden;

use crate::camera::{OrbitCamera, Pose, Projector, Ray};
use crate::editor::{Action, Editor};
use crate::panels::entities::entity_icon;
use crate::panels::filesystem::ModelDrag;
use crate::theme;

pub const SCENE_TEXTURE_ID: u64 = 1;
const ICON_PICK_RADIUS: f32 = 14.0;

#[derive(Clone, Copy, PartialEq)]
pub enum Tool {
    Select,
    Move,
    Rotate,
    Scale,
}

pub struct ViewportState {
    pub camera: OrbitCamera,
    pub tool: Tool,
    pub local_space: bool,
    pub snap: bool,
    pub show_grid: bool,
    pub show_icons: bool,
    pub show_game_ui: bool,
    ui_drag: bool,
    /// The size the viewport currently occupies, for sizing the render target.
    pub wanted_extent: [u32; 2],
    /// Set while the right mouse button is flying the camera.
    pub freelook: bool,
    /// Free-look mouse motion for this frame, supplied by the main loop.
    pub look_delta: egui::Vec2,
    /// A pick to resolve against loaded geometry.
    pub pending_pick: Option<Ray>,
    gizmo: Gizmo,
    gizmo_dragging: bool,
}

impl ViewportState {
    pub fn new() -> Self {
        Self {
            camera: OrbitCamera::default(),
            tool: Tool::Move,
            local_space: false,
            snap: false,
            show_grid: true,
            show_icons: true,
            show_game_ui: true,
            ui_drag: false,
            wanted_extent: [1280, 720],
            freelook: false,
            look_delta: egui::Vec2::ZERO,
            pending_pick: None,
            gizmo: Gizmo::default(),
            gizmo_dragging: false,
        }
    }
}

/// Where an entity is, if it has a place: from its Transform, or a light's own position.
pub fn entity_pose(world: &World, entity: EntityId) -> Option<Pose> {
    if let Ok(transform) = world.get::<&Transform>(entity) {
        return Some(Pose::from_engine(&transform));
    }
    let light = world.get::<&Pointlight>(entity).ok()?;
    let [x, y, z, _] = light.position;
    Some(Pose { position: glm::vec3(x, -y, z), rotation: glm::identity(), scale: 1.0 })
}

fn set_entity_pose(world: &World, entity: EntityId, pose: &Pose) {
    if let Ok(mut transform) = world.get::<&mut Transform>(entity) {
        pose.apply_to(&mut transform);
    } else if let Ok(mut light) = world.get::<&mut Pointlight>(entity) {
        light.position = [pose.position.x, -pose.position.y + 0.0, pose.position.z, 1.0];
    }
}

pub fn ui(ui: &mut Ui, editor: &mut Editor, world: &mut World) {
    toolbar(ui, editor, world);

    let rect = ui.available_rect_before_wrap();
    let response = ui.allocate_rect(rect, Sense::click_and_drag());
    let pixels = ui.ctx().pixels_per_point();
    editor.viewport.wanted_extent = [
        (rect.width() * pixels).round().max(16.0) as u32,
        (rect.height() * pixels).round().max(16.0) as u32,
    ];

    ui.painter().image(
        egui::TextureId::User(SCENE_TEXTURE_ID),
        rect,
        Rect::from_min_max(Pos2::ZERO, egui::pos2(1.0, 1.0)),
        Color32::WHITE,
    );
    let painter = ui.painter_at(rect);

    camera_input(ui, editor, world, &response);

    let projector = editor.viewport.camera.projector(rect);

    if editor.viewport.show_grid {
        draw_grid(&painter, &projector, &editor.viewport.camera);
    }

    // Entities without a model get an icon, so they can be seen and clicked.
    let mut icon_hits: Vec<(EntityId, Pos2)> = Vec::new();
    if editor.viewport.show_icons {
        for entity in editor.doc.order.iter().copied() {
            if world.get::<&Object3D>(entity).is_ok() || world.get::<&Hidden>(entity).is_ok() {
                continue;
            }
            let Some(pose) = entity_pose(world, entity) else {
                continue;
            };
            if let Some(pos) = projector.point(pose.position) {
                let (icon, color) = entity_icon(world, entity);
                painter.circle_filled(pos, 10.0, Color32::from_black_alpha(140));
                painter.text(pos, egui::Align2::CENTER_CENTER, icon, FontId::proportional(13.0), color);
                icon_hits.push((entity, pos));
            }
        }
    }

    if let Some(entity) = editor.doc.selection {
        if let Some(pose) = entity_pose(world, entity) {
            draw_selection(&painter, &projector, editor, world, entity, &pose);
        }
    }

    // The game's UI, as it will look in a window of this size.
    let mut ui_rects = Vec::new();
    if editor.viewport.show_game_ui {
        let mut overlay = ui.new_child(egui::UiBuilder::new().max_rect(rect));
        ui_rects = vulkano_engine::ui::draw(&mut overlay, rect, world, false).rects;
        if let Some((_, selected)) = ui_rects.iter().find(|(e, _)| Some(*e) == editor.doc.selection) {
            painter.rect_stroke(*selected, 2.0, Stroke::new(2.0, theme::selection_outline()), egui::StrokeKind::Outside);
        }
    }
    let dragged_ui = drag_ui_element(ui, editor, world, &ui_rects);

    let gizmo_busy = gizmo_ui(ui, editor, world, rect);

    // Left click selects, unless it was the gizmo's click. The nearest thing wins:
    // icons are tested here and models by the main loop against real geometry.
    //
    // This reads raw input because the gizmo places its own widget under the pointer,
    // which takes clicks away from `response`.
    let clicked = ui.input(|i| i.pointer.primary_clicked());
    if clicked && !gizmo_busy && !dragged_ui {
        let pointer = ui.input(|i| i.pointer.interact_pos());
        let on_top = pointer.is_some_and(|p| {
            rect.contains(p) && ui.ctx().layer_id_at(p) == Some(ui.layer_id())
        });
        let ui_hit = pointer
            .filter(|_| on_top)
            .and_then(|p| ui_rects.iter().rev().find(|(_, r)| r.contains(p)).map(|(e, _)| *e));
        if let Some(entity) = ui_hit {
            editor.doc.selection = Some(entity);
        } else if let Some(pointer) = pointer.filter(|_| on_top) {
            let icon = icon_hits
                .iter()
                .map(|(id, pos)| (*id, pos.distance(pointer)))
                .filter(|(_, distance)| *distance <= ICON_PICK_RADIUS)
                .min_by(|a, b| a.1.total_cmp(&b.1));
            match icon {
                Some((id, _)) => editor.doc.selection = Some(id),
                None => editor.viewport.pending_pick = Some(projector.ray(pointer)),
            }
        }
    }

    model_drop(ui, editor, &response, &projector, rect);
    overlay_text(&painter, editor, rect);
}

/// Dragging the selected UI element moves it. Returns true while it's being dragged.
fn drag_ui_element(ui: &Ui, editor: &mut Editor, world: &World, rects: &[(EntityId, Rect)]) -> bool {
    let Some(entity) = editor.doc.selection else {
        return false;
    };
    let Some((_, rect)) = rects.iter().find(|(e, _)| *e == entity) else {
        return false;
    };
    let (pressed, down, delta, origin) = ui.input(|i| {
        (i.pointer.primary_pressed(), i.pointer.primary_down(), i.pointer.delta(), i.pointer.press_origin())
    });
    if pressed {
        editor.viewport.ui_drag = origin.is_some_and(|p| rect.contains(p));
    }
    if !down {
        let was = editor.viewport.ui_drag;
        editor.viewport.ui_drag = false;
        return was;
    }
    if editor.viewport.ui_drag && delta != egui::Vec2::ZERO {
        if let Ok(mut element) = world.get::<&mut vulkano_engine::ui::UiElement>(entity) {
            let direction = element.offset_direction();
            element.x += delta.x * direction.x;
            element.y += delta.y * direction.y;
        }
    }
    editor.viewport.ui_drag
}

fn toolbar(ui: &mut Ui, editor: &mut Editor, world: &World) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        let state = &mut editor.viewport;
        let tools = [
            (Tool::Select, "⬉", "Select Tool (Q)"),
            (Tool::Move, "✚", "Move Tool (W)"),
            (Tool::Rotate, "⟳", "Rotate Tool (E)"),
            (Tool::Scale, "⛶", "Scale Tool (R)"),
        ];
        for (tool, glyph, tip) in tools {
            let button = Button::selectable(state.tool == tool, glyph).min_size(vec2(26.0, 22.0));
            if ui.add(button).on_hover_text(tip).clicked() {
                state.tool = tool;
            }
        }
        ui.add_space(8.0);
        let space = if state.local_space { "Local" } else { "Global" };
        if ui.add(Button::selectable(state.local_space, space)).on_hover_text("Gizmo orientation").clicked() {
            state.local_space = !state.local_space;
        }
        if ui.add(Button::selectable(state.snap, "Snap")).on_hover_text("Always snap (or hold Ctrl)").clicked() {
            state.snap = !state.snap;
        }
        ui.add_space(8.0);
        let focus = ui.add_enabled(editor.doc.selection.is_some(), Button::new("Focus"));
        if focus.on_hover_text("Frame the selection (F)").clicked() {
            focus_selection(editor, world);
        }
    });
}

fn camera_input(ui: &Ui, editor: &mut Editor, world: &World, response: &egui::Response) {
    let hovered = response.contains_pointer();
    let typing = ui.ctx().egui_wants_keyboard_input() || editor.is_blocked_by_dialog();
    let (pointer_delta, scroll, secondary_pressed, secondary_down, middle_down, shift, ctrl, dt) =
        ui.input(|i| {
            (
                i.pointer.delta(),
                i.raw.events.iter().map(|e| match e {
                    egui::Event::MouseWheel { delta, .. } => delta.y,
                    _ => 0.0,
                }).sum::<f32>(),
                i.pointer.button_pressed(PointerButton::Secondary),
                i.pointer.secondary_down(),
                i.pointer.middle_down(),
                i.modifiers.shift,
                i.modifiers.ctrl,
                i.stable_dt.min(0.1),
            )
        });
    let state = &mut editor.viewport;

    if hovered && secondary_pressed && !typing {
        state.freelook = true;
    }
    if !secondary_down {
        state.freelook = false;
    }

    if state.freelook {
        let look = std::mem::take(&mut state.look_delta);
        state.camera.look(look.x, look.y);
        if scroll != 0.0 {
            state.camera.fly_speed = (state.camera.fly_speed * 1.15_f32.powf(scroll.signum())).clamp(0.2, 500.0);
        }
        let camera = &state.camera;
        let (right, forward) = (camera.right(), camera.forward());
        let world_up = glm::vec3(0.0, 1.0, 0.0);
        let mut direction = glm::Vec3::zeros();
        ui.input(|i| {
            if i.key_down(Key::W) { direction += forward; }
            if i.key_down(Key::S) { direction -= forward; }
            if i.key_down(Key::D) { direction += right; }
            if i.key_down(Key::A) { direction -= right; }
            if i.key_down(Key::E) { direction += world_up; }
            if i.key_down(Key::Q) { direction -= world_up; }
        });
        if glm::length(&direction) > 0.0 {
            let speed = camera.fly_speed * if shift { 3.0 } else { 1.0 };
            state.camera.pivot += glm::normalize(&direction) * speed * dt;
            ui.ctx().request_repaint();
        }
    } else {
        if hovered && scroll != 0.0 {
            state.camera.zoom(scroll);
        }
        if middle_down && response.contains_pointer() {
            if shift {
                state.camera.pan(pointer_delta.x, pointer_delta.y);
            } else if ctrl {
                state.camera.zoom(pointer_delta.y * -0.03);
            } else {
                state.camera.orbit(pointer_delta.x, pointer_delta.y);
            }
        }
        if hovered && !typing && !ui.input(|i| i.modifiers.command) {
            ui.input(|i| {
                if i.key_pressed(Key::Q) { state.tool = Tool::Select; }
                if i.key_pressed(Key::W) { state.tool = Tool::Move; }
                if i.key_pressed(Key::E) { state.tool = Tool::Rotate; }
                if i.key_pressed(Key::R) { state.tool = Tool::Scale; }
            });
            if ui.input(|i| i.key_pressed(Key::F)) {
                focus_selection(editor, world);
            }
        }
    }
}

/// Frame the selected entity.
fn focus_selection(editor: &mut Editor, world: &World) {
    let Some(entity) = editor.doc.selection else {
        return;
    };
    let Some(pose) = entity_pose(world, entity) else {
        return;
    };
    let model = world.get::<&Object3D>(entity).ok().and_then(|o| editor.models.get(&o.model.id).cloned());
    let (center, radius) = match model {
        Some(info) => {
            let middle = (info.bounds_min + info.bounds_max) * 0.5;
            let center = pose.matrix() * glm::vec4(middle.x, middle.y, middle.z, 1.0);
            (center.xyz(), glm::length(&(info.bounds_max - info.bounds_min)) * 0.5 * pose.scale)
        }
        None => (pose.position, 1.0),
    };
    editor.viewport.camera.focus(center, radius.max(0.25));
}

/// Point the camera so the whole scene is in view.
pub fn frame_scene(editor: &mut Editor, world: &World) {
    let mut min = glm::vec3(f32::MAX, f32::MAX, f32::MAX);
    let mut max = glm::vec3(f32::MIN, f32::MIN, f32::MIN);
    for entity in editor.doc.order.iter().copied() {
        let Ok(object) = world.get::<&Object3D>(entity) else {
            continue;
        };
        let (Some(info), Some(pose)) = (editor.models.get(&object.model.id), entity_pose(world, entity)) else {
            continue;
        };
        for corner in box_corners(&pose.matrix(), info.bounds_min, info.bounds_max) {
            min = glm::min2(&min, &corner);
            max = glm::max2(&max, &corner);
        }
    }
    editor.viewport.camera = OrbitCamera::default();
    if min.x <= max.x {
        let radius = glm::length(&(max - min)) * 0.5;
        editor.viewport.camera.focus((min + max) * 0.5, radius.max(0.25));
    }
}

fn gizmo_ui(ui: &mut Ui, editor: &mut Editor, world: &World, rect: Rect) -> bool {
    let state = &mut editor.viewport;
    if state.freelook {
        return false;
    }
    let modes = match state.tool {
        Tool::Select => {
            state.gizmo_dragging = false;
            return false;
        }
        Tool::Move => GizmoMode::all_translate(),
        Tool::Rotate => GizmoMode::all_rotate(),
        // The engine only supports uniform scale, so don't offer per-axis handles.
        Tool::Scale => GizmoMode::ScaleUniform.into(),
    };
    let Some(entity) = editor.doc.selection else {
        return false;
    };
    let Some(pose) = entity_pose(world, entity) else {
        return false;
    };

    let snapping = state.snap || ui.input(|i| i.modifiers.ctrl);
    state.gizmo.update_config(GizmoConfig {
        view_matrix: row_matrix(&state.camera.view()),
        projection_matrix: row_matrix(&state.camera.projection(rect.aspect_ratio())),
        viewport: rect,
        modes,
        orientation: if state.local_space {
            GizmoOrientation::Local
        } else {
            GizmoOrientation::Global
        },
        snapping,
        snap_distance: 0.5,
        snap_angle: 15_f32.to_radians(),
        snap_scale: 0.1,
        pixels_per_point: ui.ctx().pixels_per_point(),
        ..Default::default()
    });

    let target = gizmo_transform(&pose);
    let result = state.gizmo.interact(ui, &[target]);
    let down = ui.input(|i| i.pointer.primary_down());
    let hit = result.is_some();
    if let Some((_, transforms)) = result {
        if let Some(moved) = transforms.first() {
            set_entity_pose(world, entity, &pose_from_gizmo(moved));
        }
    }
    let state = &mut editor.viewport;
    state.gizmo_dragging = (state.gizmo_dragging || hit) && down;

    // Releasing after a drag also produces a click; that must not change the selection.
    let released_drag = ui.input(|i| i.pointer.primary_released()) && hit;
    state.gizmo_dragging || hit || released_drag || state.gizmo.is_focused()
}

fn row_matrix(m: &glm::Mat4) -> mint::RowMatrix4<f64> {
    let row = |i: usize| mint::Vector4 {
        x: m[(i, 0)] as f64,
        y: m[(i, 1)] as f64,
        z: m[(i, 2)] as f64,
        w: m[(i, 3)] as f64,
    };
    mint::RowMatrix4 { x: row(0), y: row(1), z: row(2), w: row(3) }
}

fn gizmo_transform(pose: &Pose) -> GizmoTransform {
    let rotation = glm::mat3_to_quat(&glm::mat4_to_mat3(&pose.rotation));
    let scale = pose.scale as f64;
    GizmoTransform::from_scale_rotation_translation(
        mint::Vector3 { x: scale, y: scale, z: scale },
        mint::Quaternion {
            v: mint::Vector3 {
                x: rotation.i as f64,
                y: rotation.j as f64,
                z: rotation.k as f64,
            },
            s: rotation.w as f64,
        },
        mint::Vector3 {
            x: pose.position.x as f64,
            y: pose.position.y as f64,
            z: pose.position.z as f64,
        },
    )
}

fn pose_from_gizmo(transform: &GizmoTransform) -> Pose {
    let q = glm::quat(
        transform.rotation.v.x as f32,
        transform.rotation.v.y as f32,
        transform.rotation.v.z as f32,
        transform.rotation.s as f32,
    );
    Pose {
        position: glm::vec3(
            transform.translation.x as f32,
            transform.translation.y as f32,
            transform.translation.z as f32,
        ),
        rotation: glm::quat_to_mat4(&glm::quat_normalize(&q)),
        scale: transform.scale.x as f32,
    }
}

fn draw_grid(painter: &egui::Painter, projector: &Projector, camera: &OrbitCamera) {
    // Coarser cells when zoomed out, so the grid stays readable and cheap.
    let step = if camera.distance > 120.0 {
        10.0
    } else if camera.distance > 30.0 {
        5.0
    } else {
        1.0
    };
    let cells = 40;
    let extent = step * cells as f32;
    let snap = |v: f32| (v / step).round() * step;
    let (cx, cz) = (snap(camera.pivot.x), snap(camera.pivot.z));
    let minor = Color32::from_rgba_unmultiplied(255, 255, 255, 18);
    let major = Color32::from_rgba_unmultiplied(255, 255, 255, 40);

    for i in -cells..=cells {
        let x = cx + i as f32 * step;
        let z = cz + i as f32 * step;
        let is_major = |v: f32| (v / (step * 10.0)).fract().abs() < 1e-3;
        if x.abs() > 1e-3 {
            let color = if is_major(x) { major } else { minor };
            line(painter, projector, glm::vec3(x, 0.0, cz - extent), glm::vec3(x, 0.0, cz + extent), Stroke::new(1.0, color));
        }
        if z.abs() > 1e-3 {
            let color = if is_major(z) { major } else { minor };
            line(painter, projector, glm::vec3(cx - extent, 0.0, z), glm::vec3(cx + extent, 0.0, z), Stroke::new(1.0, color));
        }
    }
    // The axes through the origin, in their usual colours.
    let axis = |c: Color32| Stroke::new(1.5, c.linear_multiply(0.7));
    line(painter, projector, glm::vec3(cx - extent, 0.0, 0.0), glm::vec3(cx + extent, 0.0, 0.0), axis(theme::axis_x()));
    line(painter, projector, glm::vec3(0.0, 0.0, cz - extent), glm::vec3(0.0, 0.0, cz + extent), axis(theme::axis_z()));
}

fn line(painter: &egui::Painter, projector: &Projector, a: glm::Vec3, b: glm::Vec3, stroke: Stroke) {
    if let Some(points) = projector.line(a, b) {
        painter.line_segment(points, stroke);
    }
}

fn draw_selection(painter: &egui::Painter, projector: &Projector, editor: &Editor, world: &World, entity: EntityId, pose: &Pose) {
    if let Ok(object) = world.get::<&Object3D>(entity) {
        if let Some(info) = editor.models.get(&object.model.id) {
            draw_box_corners(painter, projector, &pose.matrix(), info.bounds_min, info.bounds_max);
        }
    }
    if let Ok(light) = world.get::<&Pointlight>(entity) {
        draw_circles(painter, projector, pose, light.radius, Stroke::new(1.0, theme::warning().linear_multiply(0.7)));
    }
    if let Ok(collider) = world.get::<&ColliderComponent>(entity) {
        let stroke = Stroke::new(1.5, theme::collider());
        // Colliders don't scale with the transform in the physics engine.
        let unscaled = Pose { scale: 1.0, ..*pose };
        if let Some(cuboid) = collider.shape.as_cuboid() {
            let half = cuboid.half_extents;
            draw_box(painter, projector, &unscaled.matrix(), glm::vec3(-half.x, -half.y, -half.z), glm::vec3(half.x, half.y, half.z), stroke);
        } else if let Some(ball) = collider.shape.as_ball() {
            draw_circles(painter, projector, &unscaled, ball.radius, stroke);
        } else if let Some(capsule) = collider.shape.as_capsule() {
            let up = (unscaled.rotation * glm::vec4(0.0, capsule.half_height(), 0.0, 0.0)).xyz();
            for center in [unscaled.position + up, unscaled.position - up] {
                draw_circles(painter, projector, &Pose { position: center, ..unscaled }, capsule.radius, stroke);
            }
        }
    }
}

fn box_corners(matrix: &glm::Mat4, min: glm::Vec3, max: glm::Vec3) -> [glm::Vec3; 8] {
    let mut corners = [glm::Vec3::zeros(); 8];
    for (i, corner) in corners.iter_mut().enumerate() {
        let x = if i & 1 == 0 { min.x } else { max.x };
        let y = if i & 2 == 0 { min.y } else { max.y };
        let z = if i & 4 == 0 { min.z } else { max.z };
        *corner = (matrix * glm::vec4(x, y, z, 1.0)).xyz();
    }
    corners
}

const BOX_EDGES: [(usize, usize); 12] = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
];

fn draw_box(painter: &egui::Painter, projector: &Projector, matrix: &glm::Mat4, min: glm::Vec3, max: glm::Vec3, stroke: Stroke) {
    let corners = box_corners(matrix, min, max);
    for (a, b) in BOX_EDGES {
        line(painter, projector, corners[a], corners[b], stroke);
    }
}

/// Godot's selection look: only the corners of the bounding box.
fn draw_box_corners(painter: &egui::Painter, projector: &Projector, matrix: &glm::Mat4, min: glm::Vec3, max: glm::Vec3) {
    let corners = box_corners(matrix, min, max);
    let stroke = Stroke::new(2.0, theme::selection_outline());
    for (a, b) in BOX_EDGES {
        let (start, end) = (corners[a], corners[b]);
        line(painter, projector, start, start + (end - start) * 0.2, stroke);
        line(painter, projector, end, end + (start - end) * 0.2, stroke);
    }
}

fn draw_circles(painter: &egui::Painter, projector: &Projector, pose: &Pose, radius: f32, stroke: Stroke) {
    const SEGMENTS: usize = 40;
    for plane in 0..3 {
        let point = |i: usize| {
            let angle = i as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
            let (s, c) = angle.sin_cos();
            let local = match plane {
                0 => glm::vec4(c, s, 0.0, 0.0),
                1 => glm::vec4(c, 0.0, s, 0.0),
                _ => glm::vec4(0.0, c, s, 0.0),
            } * radius;
            pose.position + (pose.rotation * local).xyz()
        };
        for i in 0..SEGMENTS {
            line(painter, projector, point(i), point(i + 1), stroke);
        }
    }
}

fn model_drop(ui: &Ui, editor: &mut Editor, response: &egui::Response, projector: &Projector, rect: Rect) {
    if response.dnd_hover_payload::<ModelDrag>().is_some() {
        ui.painter_at(rect).rect_stroke(rect.shrink(1.0), 0.0, Stroke::new(2.0, theme::accent()), egui::StrokeKind::Inside);
    }
    if let Some(payload) = response.dnd_release_payload::<ModelDrag>() {
        let position = ui
            .input(|i| i.pointer.interact_pos())
            .and_then(|p| projector.ray(p).ground_point())
            .map(|p| [p.x, p.y, p.z]);
        let path = payload.0.clone();
        if path.ends_with(".vscene") {
            editor.actions.push(Action::AddPrefab { path, position });
        } else {
            editor.actions.push(Action::AddModel { path, position });
        }
    }
}

fn overlay_text(painter: &egui::Painter, editor: &Editor, rect: Rect) {
    let font = FontId::proportional(12.0);
    let shadowed = |pos: Pos2, align: egui::Align2, text: String, color: Color32| {
        painter.text(pos + vec2(1.0, 1.0), align, &text, font.clone(), Color32::BLACK);
        painter.text(pos, align, text, font.clone(), color);
    };
    shadowed(rect.left_top() + vec2(8.0, 6.0), egui::Align2::LEFT_TOP, "Perspective".to_string(), theme::text());
    let fps = if editor.frame_ms > 0.0 { 1000.0 / editor.frame_ms } else { 0.0 };
    shadowed(
        rect.right_top() + vec2(-8.0, 6.0),
        egui::Align2::RIGHT_TOP,
        format!("{fps:.0} FPS  {:.1} ms", editor.frame_ms),
        theme::text_dim(),
    );
    if editor.viewport.freelook {
        let speed = format!("Fly speed: {:.1} m/s (wheel to change)", editor.viewport.camera.fly_speed);
        shadowed(rect.left_bottom() + vec2(8.0, -8.0), egui::Align2::LEFT_BOTTOM, speed, theme::accent());
    }
}
