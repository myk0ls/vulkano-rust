pub mod mesh;
pub mod model;
pub mod renderer;
pub mod skybox;

use shipyard::World;

pub fn run_graphics_systems(world: &mut World) {}

// pub fn update_camera_view(cameras: View<Camera>, transforms: View<Transform>) {
//     for (camera, transform) in (&cameras, &transforms).iter().filter(|(c, _)| c.active) {
//         //FPS camera: look from position in the direction we're facing
//         let pos = transform.get_position_vector();
//         let position = vec3(pos[0], pos[1], pos[2]);

//         let forward = camera.get_forward_vector();
//         let target = position + forward; // Look ahead from our position
//         let up = vec3(0.0, 1.0, 0.0);

//         let view = look_at(&position, &target, &up);
//         self.renderer.set_view(&view);
//         break;
//     }
// }
