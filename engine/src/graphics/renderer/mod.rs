mod compute;
mod ibl;
mod images;
mod passes;
mod custom;
mod pipelines;
pub use custom::{PostParam, check_material, check_post_effect, parse_post_params};
mod upload;

use std::{mem, sync::Arc};

use nalgebra_glm::{
    TMat4, TVec3, half_pi, identity, inverse, look_at_rh, ortho, perspective, vec3,
};
use sdl3::video::Window;
use vulkano::command_buffer::{DrawIndexedIndirectCommand, PrimaryCommandBufferAbstract};
use vulkano::{
    Handle, Validated, Version, VulkanError, VulkanLibrary, VulkanObject,
    buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo, allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageUsage,
        sampler::{
            BorderColor, Filter, LOD_CLAMP_NONE, Sampler, SamplerAddressMode, SamplerCreateInfo,
            SamplerMipmapMode,
        },
        view::{ImageView, ImageViewCreateInfo},
    },
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        graphics::viewport::{Viewport, ViewportState},
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        PresentMode, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo,
        SwapchainPresentInfo, acquire_next_image,
    },
    sync::{self, GpuFuture},
};

use ash::vk;

use nalgebra_glm::TVec3 as Vec3;

use crate::assets::asset_manager::{self, UnifiedGeometry};
use crate::assets::gltf_loader::DummyVertex;
use crate::scene::components::pointlight::Pointlight;
use crate::scene::scene_file::Environment;
use crate::scene::components::transform::Transform;

// ── Shader modules ─────────────────────────────────────────────────────────

pub(crate) mod deferred_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/graphics/renderer/shaders/deferred.vert",
    }
}
pub(crate) mod deferred_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/graphics/renderer/shaders/deferred.frag",
    }
}
mod directional_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/graphics/renderer/shaders/directional.vert"
    }
}
mod directional_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/graphics/renderer/shaders/directional.frag",
    }
}
mod ambient_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/graphics/renderer/shaders/ambient.vert"
    }
}
mod ambient_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/graphics/renderer/shaders/ambient.frag",
    }
}
mod light_obj_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/graphics/renderer/shaders/light_obj.vert",
    }
}
mod light_obj_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/graphics/renderer/shaders/light_obj.frag",
    }
}
mod skybox_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/graphics/renderer/shaders/skybox.vert",
    }
}
mod skybox_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/graphics/renderer/shaders/skybox.frag",
    }
}
mod pointlight_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/graphics/renderer/shaders/pointlight.vert",
    }
}
mod clustered_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/graphics/renderer/shaders/clustered.frag",
    }
}
mod light_cluster_comp {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/graphics/renderer/shaders/light_cluster.comp",
    }
}
mod shadows_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/graphics/renderer/shaders/shadows.vert",
    }
}
mod shadows_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/graphics/renderer/shaders/shadows.frag",
    }
}
mod ao_comp {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/graphics/renderer/shaders/ao.comp"
    }
}
mod blur_comp {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/graphics/renderer/shaders/blur.comp"
    }
}
mod fxaa_comp {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/graphics/renderer/shaders/fxaa.comp"
    }
}
mod composite_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/graphics/renderer/shaders/composite.vert",
    }
}
mod composite_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/graphics/renderer/shaders/composite.frag",
    }
}
mod cull_comp {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/graphics/renderer/shaders/frustum_culling.comp",
    }
}
mod irradiance_comp {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/graphics/renderer/shaders/irradiance.comp",
    }
}
mod prefilter_comp {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/graphics/renderer/shaders/prefilter.comp",
    }
}
mod brdf_lut_comp {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/graphics/renderer/shaders/brdf_lut.comp",
    }
}

// ── Public re-exports (was in old mod.rs) ──────────────────────────────────

const SHADOW_MAP_SIZE: u32 = 4096;
/// Width in metres of the square the sun's shadow map covers around the camera. Larger
/// reaches farther but makes shadows blurrier. directional.frag assumes this value.
const SUN_SHADOW_EXTENT: f32 = 50.0;
/// How far along the sun's direction the shadow map reaches, centred on the camera's area,
/// so tall casters well above the player still cast into view.
const SUN_SHADOW_DEPTH: f32 = 200.0;
/// Size of each local light (spot or cube face) shadow map.
const LOCAL_SHADOW_SIZE: u32 = 1024;
/// How many spot lights and point lights can cast shadows at once. Each spot light costs
/// one extra render of the scene's geometry, each point light six.
pub const MAX_SPOT_SHADOWS: usize = 4;
pub const MAX_POINT_SHADOWS: usize = 2;
const LOCAL_SHADOW_LAYERS: u32 = (MAX_SPOT_SHADOWS + 6 * MAX_POINT_SHADOWS) as u32;

/// Clustered lighting: the view frustum is split into this many tiles across, down, and
/// in depth. Must match `GRID` in light_cluster.comp and clustered.frag.
const CLUSTER_GRID: [u32; 3] = [16, 9, 24];
const CLUSTER_COUNT: u32 = CLUSTER_GRID[0] * CLUSTER_GRID[1] * CLUSTER_GRID[2];
/// u32s per cluster: a count and up to 127 light indices. Must match `SLOT` in the shaders.
const CLUSTER_SLOT: u32 = 128;
/// Depth slices are spaced exponentially between these distances; the first slice also
/// covers everything nearer and the last everything farther.
const CLUSTER_NEAR: f32 = 0.1;
const CLUSTER_FAR: f32 = 200.0;
/// The most point and spot lights drawn in one frame.
pub const MAX_LOCAL_LIGHTS: usize = 4096;

/// This frame's lights and their cluster lists, from `cluster_lights` for the lighting pass.
pub(super) struct ClusterFrame {
    pub lights: Subbuffer<[clustered_frag::Light]>,
    pub clusters: Subbuffer<[u32]>,
    pub frame: Subbuffer<clustered_frag::Frame_Data>,
    pub shadow: Subbuffer<clustered_frag::Shadow_Data>,
}

/// A point or spot light, in engine space, as the renderer draws it.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalLight {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
    /// For spot lights: the direction it shines, and the inner and outer cone angles in
    /// degrees.
    pub spot: Option<([f32; 3], f32, f32)>,
    pub cast_shadows: bool,
}

/// Where a local light's shadow is stored this frame.
#[derive(Clone, Debug)]
pub(crate) struct ShadowSlot {
    pub layer: u32,
    pub matrices: Vec<TMat4<f32>>,
}

#[derive(Debug, Clone)]
enum RenderStage {
    Stopped,
    Deferred,
    Ambient,
    Directional,
    LightObject,
    NeedsRedraw,
    Shadow,
}

#[derive(Debug, Clone)]
struct VP {
    view: TMat4<f32>,
    projection: TMat4<f32>,
    camera_pos: TVec3<f32>,
}

impl VP {
    fn new() -> VP {
        VP {
            view: identity(),
            projection: identity(),
            camera_pos: vec3(0.0, 0.0, 0.0),
        }
    }
}

/// One draw call's worth of a frame's draws, all using the same material shader.
pub struct DrawRange {
    pub shader: Option<String>,
    pub start: u32,
    pub count: u32,
}

/// What `cull_pass` needs to know about one draw.
pub struct DrawItem {
    pub draw_index: usize,
    pub transform: Transform,
    pub skin_offset: u32,
    pub override_index: u32,
    /// A material shader file, or None for the standard material.
    pub shader: Option<String>,
}

pub struct CulledDrawBuffers {
    pub overrides: Subbuffer<[asset_manager::GpuOverride]>,
    pub ranges: Vec<DrawRange>,
    /// Draws for the camera; the cull pass zeroes the instance count of what it can't see.
    pub indirect: Subbuffer<[DrawIndexedIndirectCommand]>,
    /// Every draw, unculled, for the shadow passes: an object off screen can still cast
    /// a shadow into view.
    pub shadow_indirect: Subbuffer<[DrawIndexedIndirectCommand]>,
    pub draw_data: Subbuffer<[asset_manager::DrawData]>,
    pub joint_matrices: Subbuffer<[[[f32; 4]; 4]]>,
}

// ── Renderer struct ────────────────────────────────────────────────────────

pub struct Renderer {
    pub(super) instance: Arc<Instance>,
    pub(super) surface: Arc<Surface>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub(super) vp: VP,
    pub(super) swapchain: Arc<Swapchain>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub(super) descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub(super) vp_buffer: Subbuffer<deferred_vert::VP_Data>,
    pub(super) ambient_buffer: Subbuffer<ambient_frag::Ambient_Data>,
    pub(super) directional_subbuffer: Subbuffer<directional_frag::Directional_Light_Data>,
    pub(super) directional_allocator: SubbufferAllocator,
    pub(super) frag_location_buffer: Arc<ImageView>,
    pub(super) specular_buffer: Arc<ImageView>,
    pub(super) sampler: Arc<Sampler>,
    pub(super) clamp_sampler: Arc<Sampler>,
    pub(super) shadow_sampler: Arc<Sampler>,
    pub(super) shadow_map_view: Arc<ImageView>,
    pub(super) render_pass: Arc<RenderPass>,
    pub(super) shadow_render_pass: Arc<RenderPass>,
    pub(super) shadow_pipeline: Arc<GraphicsPipeline>,
    pub(super) deferred_pipeline: Arc<GraphicsPipeline>,
    pub(super) directional_pipeline: Arc<GraphicsPipeline>,
    /// Shades all point and spot lights in one pass, using the cluster lists.
    pub(super) clustered_pipeline: Arc<GraphicsPipeline>,
    /// Builds the cluster lists.
    pub(super) light_cluster_pipeline: Arc<ComputePipeline>,
    pub(super) cluster_frame: Option<ClusterFrame>,
    pub(super) ambient_pipeline: Arc<GraphicsPipeline>,
    pub(super) light_obj_pipeline: Arc<GraphicsPipeline>,
    pub(super) skybox_pipeline: Arc<GraphicsPipeline>,
    pub(super) ao_pipeline: Arc<ComputePipeline>,
    pub(super) blur_pipeline: Arc<ComputePipeline>,
    pub(super) fxaa_pipeline: Arc<ComputePipeline>,
    pub(super) cull_pipeline: Arc<ComputePipeline>,
    pub(super) composite_pipeline: Arc<GraphicsPipeline>,
    pub(super) composite_render_pass: Arc<RenderPass>,
    pub(super) composite_framebuffers: Vec<Arc<Framebuffer>>,
    pub(super) scene_image: Arc<ImageView>,
    pub(super) dummy_verts: Subbuffer<[DummyVertex]>,
    pub(super) framebuffers: Vec<Arc<Framebuffer>>,
    pub(super) shadow_framebuffer: Arc<Framebuffer>,
    pub(super) color_buffer: Arc<ImageView>,
    pub(super) normal_buffer: Arc<ImageView>,
    pub(super) depth_buffer: Arc<ImageView>,
    pub(super) ao_image: Arc<ImageView>,
    pub(super) ao_blurred_image: Arc<ImageView>,
    pub(super) fxaa_image: Arc<ImageView>,
    pub(super) ao_sampler: Arc<Sampler>,
    pub(super) ao_repeat_sampler: Arc<Sampler>,
    pub(super) ao_rotation_image: Arc<ImageView>,
    pub(super) vp_set: Arc<DescriptorSet>,
    pub(super) bindless_material_set: Option<Arc<DescriptorSet>>,
    pub(super) viewport: Viewport,
    pub(super) render_stage: RenderStage,
    pub commands: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    pub(super) image_index: u32,
    pub(super) acquire_future: Option<SwapchainAcquireFuture>,
    pub ao_radius: f32,
    pub ao_att_scale: f32,
    pub ao_dist_scale: f32,
    pub ao_blur_depth_threshold: f32,
    pub ao_composite_scale: f32,
    pub ao_composite_bias: f32,
    pub exposure: f32,
    pub fxaa_enabled: bool,
    pub shadow_softness: f32,

    pub(super) composite_format: Format,
    pub(super) local_shadow_view: Arc<ImageView>,
    pub(super) local_shadow_framebuffers: Vec<Arc<Framebuffer>>,
    pub(super) local_lights: Vec<LocalLight>,
    pub(super) local_slots: Vec<Option<ShadowSlot>>,
    /// Whether the sun casts shadows.
    pub sun_shadows: bool,
    pub(super) custom: custom::CustomShaders,
    pub(super) post_effects: Vec<crate::scene::scene_file::PostEffect>,
    pub(super) post_images: Option<[Arc<ImageView>; 2]>,
    /// The image composite reads this frame: the last post effect's output, if any ran.
    pub(super) post_output: Option<Arc<ImageView>>,
    pub(super) start_time: std::time::Instant,
    pub(super) offscreen: Option<Arc<ImageView>>,
    pub(super) swapchain_images: Vec<Arc<Image>>,
    pub(super) swapchain_generation: u64,

    // Declared last so the window outlives the Vulkan surface on drop.
    pub(super) window: Window,
}

// ── Core impl ─────────────────────────────────────────────────────────────

/// Options for applications that need more than the default game setup.
#[derive(Clone)]
pub struct RendererConfig {
    pub present_mode: PresentMode,
    /// Swapchain formats to try first; falls back to the surface's first format.
    pub preferred_swapchain_formats: Vec<Format>,
    pub swapchain_usage: ImageUsage,
    /// Format the scene is composited to. Defaults to the swapchain format.
    pub composite_format: Option<Format>,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            present_mode: PresentMode::Immediate,
            // The composite shader relies on the target applying gamma encoding.
            preferred_swapchain_formats: vec![Format::B8G8R8A8_SRGB, Format::R8G8B8A8_SRGB],
            // TRANSFER_SRC allows screenshots.
            swapchain_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            composite_format: None,
        }
    }
}

impl Renderer {
    pub fn new(window: &Window) -> Renderer {
        Self::with_config(window, RendererConfig::default())
    }

    pub fn with_config(window: &Window, config: RendererConfig) -> Renderer {
        let instance = {
            let library = VulkanLibrary::new().unwrap();
            let sdl_extensions = window.vulkan_instance_extensions().unwrap();
            let extensions: InstanceExtensions =
                sdl_extensions.iter().map(|s| s.as_str()).collect();
            Instance::new(
                library,
                InstanceCreateInfo {
                    enabled_extensions: extensions,
                    max_api_version: Some(Version::V1_3),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let raw_instance: vk::Instance = instance.handle();
        let raw_instance_ptr = raw_instance.as_raw() as *mut vk::Instance;
        let raw_surface_ptr = window.vulkan_create_surface(raw_instance_ptr as _).unwrap();
        let raw_surface = vk::SurfaceKHR::from_raw(raw_surface_ptr as u64);

        let surface = unsafe {
            Arc::new(Surface::from_handle(
                Arc::clone(&instance),
                raw_surface,
                vulkano::swapchain::SurfaceApi::Win32,
                None,
            ))
        };

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ext_descriptor_indexing: true,
            ..DeviceExtensions::empty()
        };

        let device_features = DeviceFeatures {
            descriptor_binding_partially_bound: true,
            runtime_descriptor_array: true,
            descriptor_binding_variable_descriptor_count: true,
            sampler_anisotropy: true,
            shader_sampled_image_array_non_uniform_indexing: true,
            multi_draw_indirect: true,
            shader_draw_parameters: true,
            ..DeviceFeatures::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("No suitable physical device found");

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: device_features,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let mut vp = VP::new();

        let (swapchain, images) = {
            let caps = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
            let surface_formats = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap();
            let image_format = config
                .preferred_swapchain_formats
                .iter()
                .copied()
                .find(|wanted| surface_formats.iter().any(|(f, _)| f == wanted))
                .or(Some(surface_formats[0].0));
            let image_extent: [u32; 2] = window.size().into();
            let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
            vp.projection = perspective(aspect_ratio, half_pi(), 0.01, 1000.0);

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: caps.min_image_count,
                    image_format: image_format.unwrap(),
                    image_extent,
                    image_usage: config.swapchain_usage,
                    composite_alpha: alpha,
                    present_mode: config.present_mode,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let composite_format = config.composite_format.unwrap_or(swapchain.image_format());

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                update_after_bind: true,
                set_count: 32,
                ..Default::default()
            },
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // ── Render passes ──────────────────────────────────────────────────

        let shadow_render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                depth: {
                    format: Format::D32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [],
                depth_stencil: {depth},
            }
        )
        .unwrap();

        let render_pass = vulkano::ordered_passes_renderpass!(device.clone(),
            attachments: {
                final_color: {
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                color: {
                    format: Format::A2B10G10R10_UNORM_PACK32,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
                normals: {
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                frag_location: {
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
                specular: {
                    format: Format::R16G16_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
                depth: {
                    format: Format::D32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                }
            },
            passes: [
                {
                    color: [color, normals, frag_location, specular, final_color],
                    depth_stencil: {depth},
                    input: []
                },
                {
                    color: [final_color],
                    depth_stencil: {depth},
                    input: [color, normals, frag_location, specular]
                }
            ]
        )
        .unwrap();

        let mut viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [
                swapchain.image_extent()[0] as f32,
                swapchain.image_extent()[1] as f32,
            ],
            depth_range: 0.0..=1.0,
        };

        let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
        let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

        let composite_render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: composite_format,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            }
        )
        .unwrap();

        let composite_pass = Subpass::from(composite_render_pass.clone(), 0).unwrap();
        let shadow_pass = Subpass::from(shadow_render_pass.clone(), 0).unwrap();

        // ── Pipelines ──────────────────────────────────────────────────────

        let deferred_pipeline =
            pipelines::create_deferred(device.clone(), viewport.clone(), deferred_pass);
        let directional_pipeline =
            pipelines::create_directional(device.clone(), viewport.clone(), lighting_pass.clone());
        let clustered_pipeline =
            pipelines::create_clustered(device.clone(), viewport.clone(), lighting_pass.clone());
        let light_cluster_pipeline = pipelines::create_light_cluster(device.clone());
        let ambient_pipeline =
            pipelines::create_ambient(device.clone(), viewport.clone(), lighting_pass.clone());
        let light_obj_pipeline =
            pipelines::create_light_obj(device.clone(), viewport.clone(), lighting_pass.clone());
        let skybox_pipeline =
            pipelines::create_skybox(device.clone(), viewport.clone(), lighting_pass.clone());
        let shadow_pipeline = pipelines::create_shadow(device.clone(), shadow_pass);
        let ao_pipeline = pipelines::create_ao(device.clone());
        let blur_pipeline = pipelines::create_blur(device.clone());
        let fxaa_pipeline = pipelines::create_fxaa(device.clone());
        let cull_pipeline = pipelines::create_cull(device.clone());
        let composite_pipeline =
            pipelines::create_composite(device.clone(), viewport.clone(), composite_pass);

        // ── Buffers ────────────────────────────────────────────────────────

        let vp_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            deferred_vert::VP_Data {
                view: vp.view.into(),
                projection: vp.projection.into(),
            },
        )
        .unwrap();

        let ambient_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            ambient_frag::Ambient_Data {
                color: [1.0, 1.0, 1.0],
                intensity: 0.1,
            },
        )
        .unwrap();

        let directional_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let directional_subbuffer: Subbuffer<directional_frag::Directional_Light_Data> =
            directional_allocator.allocate_sized().unwrap();

        let dummy_verts = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DummyVertex::list().iter().cloned(),
        )
        .unwrap();

        // ── Window-size images and framebuffers ────────────────────────────

        // The swapchain can't be a composite target if its format differs, so such
        // setups render off-screen from the start.
        let mut offscreen = None;
        let initial_targets = if composite_format == swapchain.image_format() {
            images.clone()
        } else {
            let [width, height] = swapchain.image_extent();
            let image = Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: composite_format,
                    extent: [width, height, 1],
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();
            offscreen = Some(ImageView::new_default(image.clone()).unwrap());
            vec![image]
        };

        let (
            framebuffers,
            composite_framebuffers,
            scene_image,
            color_buffer,
            normal_buffer,
            frag_location_buffer,
            specular_buffer,
            depth_buffer,
            ao_image,
            ao_blurred_image,
            fxaa_image,
        ) = Renderer::window_size_dependent_setup(
            memory_allocator.clone(),
            &initial_targets,
            render_pass.clone(),
            composite_render_pass.clone(),
            composite_format,
            &mut viewport,
        );

        // ── Shadow map ─────────────────────────────────────────────────────

        let shadow_map_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: Format::D32_SFLOAT,
                extent: [SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        let shadow_map_view = ImageView::new_default(shadow_map_image).unwrap();

        let shadow_framebuffer = Framebuffer::new(
            shadow_render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![shadow_map_view.clone()],
                ..Default::default()
            },
        )
        .unwrap();

        // ── Local light shadow maps: one layer per spot light or cube face ──

        let local_shadow_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: Format::D32_SFLOAT,
                extent: [LOCAL_SHADOW_SIZE, LOCAL_SHADOW_SIZE, 1],
                array_layers: LOCAL_SHADOW_LAYERS,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();
        let local_shadow_view = ImageView::new(
            local_shadow_image.clone(),
            ImageViewCreateInfo {
                view_type: vulkano::image::view::ImageViewType::Dim2dArray,
                ..ImageViewCreateInfo::from_image(&local_shadow_image)
            },
        )
        .unwrap();
        let local_shadow_framebuffers = (0..LOCAL_SHADOW_LAYERS)
            .map(|layer| {
                let mut info = ImageViewCreateInfo::from_image(&local_shadow_image);
                info.view_type = vulkano::image::view::ImageViewType::Dim2d;
                info.subresource_range.array_layers = layer..layer + 1;
                let view = ImageView::new(local_shadow_image.clone(), info).unwrap();
                Framebuffer::new(
                    shadow_render_pass.clone(),
                    FramebufferCreateInfo { attachments: vec![view], ..Default::default() },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        // ── Samplers ───────────────────────────────────────────────────────

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 0.0,
                lod: 0.0..=LOD_CLAMP_NONE,
                anisotropy: Some(16.0),
                ..Default::default()
            },
        )
        .unwrap();

        let clamp_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                lod: 0.0..=LOD_CLAMP_NONE,
                ..Default::default()
            },
        )
        .unwrap();

        let shadow_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                mip_lod_bias: 0.0,
                compare: Some(vulkano::pipeline::graphics::depth_stencil::CompareOp::LessOrEqual),
                address_mode: [SamplerAddressMode::ClampToBorder; 3],
                border_color: BorderColor::FloatOpaqueWhite,
                ..Default::default()
            },
        )
        .unwrap();

        let ao_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let ao_repeat_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        // ── AO rotation texture ────────────────────────────────────────────

        #[rustfmt::skip]
        let rotation_pixels: [u8; 64] = [
            148, 203,  56, 255,   12,  87, 234, 255,  199,  41, 122, 255,   77, 166,  33, 255,
             91, 218, 174, 255,  233,  14,  67, 255,  155, 139, 211, 255,   38, 249,  98, 255,
            177,  62, 185, 255,  104, 191,  19, 255,   23, 133, 247, 255,  212,  78, 141, 255,
             59, 222,  88, 255,  167,   5, 196, 255,  240, 112,  45, 255,   83, 158, 229, 255,
        ];

        let rotation_upload = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            rotation_pixels,
        )
        .unwrap();

        let rotation_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [4, 4, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        {
            let mut upload_cmd = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            upload_cmd
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    rotation_upload,
                    rotation_image.clone(),
                ))
                .unwrap();
            upload_cmd
                .build()
                .unwrap()
                .execute(queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
        }

        let ao_rotation_image = ImageView::new_default(rotation_image).unwrap();

        // ── VP descriptor set ──────────────────────────────────────────────

        let vp_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();
        let vp_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, vp_buffer.clone())],
            [],
        )
        .unwrap();

        Renderer {
            instance,
            surface,
            device,
            queue,
            vp,
            swapchain,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            vp_buffer,
            ambient_buffer,
            directional_subbuffer,
            directional_allocator,
            frag_location_buffer,
            specular_buffer,
            sampler,
            clamp_sampler,
            shadow_sampler,
            shadow_map_view,
            render_pass,
            shadow_render_pass,
            shadow_pipeline,
            deferred_pipeline,
            directional_pipeline,
            clustered_pipeline,
            light_cluster_pipeline,
            cluster_frame: None,
            ambient_pipeline,
            light_obj_pipeline,
            skybox_pipeline,
            ao_pipeline,
            blur_pipeline,
            fxaa_pipeline,
            cull_pipeline,
            composite_pipeline,
            composite_render_pass,
            composite_framebuffers,
            scene_image,
            dummy_verts,
            framebuffers,
            shadow_framebuffer,
            color_buffer,
            normal_buffer,
            depth_buffer,
            ao_image,
            ao_blurred_image,
            fxaa_image,
            ao_sampler,
            ao_repeat_sampler,
            ao_rotation_image,
            vp_set,
            bindless_material_set: None,
            viewport,
            render_stage: RenderStage::Stopped,
            commands: None,
            image_index: 0,
            acquire_future: None,
            ao_radius: 0.05,
            ao_att_scale: 0.95,
            ao_dist_scale: 1.7,
            ao_blur_depth_threshold: 100.0,
            ao_composite_scale: 1.0,
            ao_composite_bias: 0.0,
            exposure: 1.0,
            fxaa_enabled: true,
            shadow_softness: 2.0,
            composite_format,
            local_shadow_view,
            local_shadow_framebuffers,
            local_lights: Vec::new(),
            local_slots: Vec::new(),
            sun_shadows: true,
            custom: Default::default(),
            post_effects: Vec::new(),
            post_images: None,
            post_output: None,
            start_time: std::time::Instant::now(),
            offscreen,
            swapchain_images: images,
            swapchain_generation: 0,
            window: window.clone(),
        }
    }

    pub fn start(&mut self) {
        match self.render_stage {
            RenderStage::Stopped => {
                self.render_stage = RenderStage::Deferred;
            }
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(Validated::Error(VulkanError::OutOfDate)) => {
                    self.recreate_swapchain();
                    return;
                }
                Err(err) => panic!("{:?}", err),
            };

        if suboptimal {
            self.recreate_swapchain();
            return;
        }

        let commands = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        self.commands = Some(commands);
        self.image_index = image_index;
        self.acquire_future = Some(acquire_future);
    }

    pub fn finish(&mut self, previous_frame_end: &mut Option<Box<dyn GpuFuture>>) {
        self.finish_with(previous_frame_end, |_, _| {});
    }

    /// True between a successful `start` and `finish`. False when this frame was
    /// skipped, for example because the swapchain had to be recreated.
    pub fn frame_active(&self) -> bool {
        self.commands.is_some()
    }

    /// Like `finish`, but lets the caller record extra commands (such as a UI pass)
    /// after the scene is composited. The callback receives the swapchain image
    /// index and only runs for frames that are actually presented.
    pub fn finish_with(
        &mut self,
        previous_frame_end: &mut Option<Box<dyn GpuFuture>>,
        overlay: impl FnOnce(&mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, u32),
    ) {
        match self.render_stage {
            RenderStage::Directional => {}
            RenderStage::LightObject => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let mut commands = self.commands.take().unwrap();
        commands.end_render_pass(SubpassEndInfo::default()).unwrap();

        self.dispatch_ao(&mut commands);
        self.dispatch_blur(&mut commands);
        self.dispatch_fxaa(&mut commands);
        self.dispatch_post_effects(&mut commands);
        self.composite(&mut commands);
        overlay(&mut commands, self.image_index);

        let command_buffer = commands.build().unwrap();
        let af = self.acquire_future.take().unwrap();

        let mut local_future: Option<Box<dyn GpuFuture>> =
            Some(Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>);
        mem::swap(&mut local_future, previous_frame_end);

        let future = local_future
            .take()
            .unwrap()
            .join(af)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    self.image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                // Wait for the GPU so frames can't queue up ahead of it; the next
                // frame then samples input as late as possible, reducing input lag.
                if let Err(e) = future.wait(None) {
                    println!("Failed to wait for frame: {:?}", e);
                }
                *previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(Validated::Error(VulkanError::OutOfDate)) => {
                self.recreate_swapchain();
                *previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                *previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
        }

        self.commands = None;
        self.render_stage = RenderStage::Stopped;
    }

    fn begin_main_render_pass(&mut self) {
        // Light clustering is a compute dispatch, which can't run inside a render pass.
        self.cluster_lights();
        let clear_values = vec![
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0].into()),
            Some(1.0.into()),
        ];
        let framebuffer = self.framebuffers[self.target_index()].clone();

        self.commands
            .as_mut()
            .unwrap()
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values,
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();
    }

    pub fn swapchain_images(&self) -> &[Arc<Image>] {
        &self.swapchain_images
    }

    pub fn swapchain_format(&self) -> Format {
        self.swapchain.image_format()
    }

    pub fn swapchain_extent(&self) -> [u32; 2] {
        self.swapchain.image_extent()
    }

    /// Incremented whenever the swapchain images are replaced.
    pub fn swapchain_generation(&self) -> u64 {
        self.swapchain_generation
    }

    pub fn descriptor_set_allocator(&self) -> Arc<StandardDescriptorSetAllocator> {
        self.descriptor_set_allocator.clone()
    }

    /// Post effects to run after anti-aliasing, in order. Cheap to call every frame.
    pub fn set_post_effects(&mut self, effects: &[crate::scene::scene_file::PostEffect]) {
        if self.post_effects.as_slice() != effects {
            self.post_effects = effects.to_vec();
        }
    }

    /// The point and spot lights for this frame, before its shadow pass. Shadows go to the
    /// first lights that ask for them, up to the limits.
    pub fn set_local_lights(&mut self, lights: Vec<LocalLight>) {
        let mut spots = 0;
        let mut points = 0;
        self.local_slots = lights
            .iter()
            .map(|light| {
                if !light.cast_shadows {
                    return None;
                }
                match light.spot {
                    Some((direction, _, outer)) if spots < MAX_SPOT_SHADOWS => {
                        spots += 1;
                        let layer = (spots - 1) as u32;
                        Some(ShadowSlot { layer, matrices: vec![spot_matrix(light.position, direction, outer, light.radius)] })
                    }
                    None if points < MAX_POINT_SHADOWS => {
                        points += 1;
                        let layer = (MAX_SPOT_SHADOWS + 6 * (points - 1)) as u32;
                        Some(ShadowSlot { layer, matrices: cube_matrices(light.position, light.radius) })
                    }
                    _ => None,
                }
            })
            .collect();
        self.local_lights = lights;
        self.local_lights.truncate(MAX_LOCAL_LIGHTS);
        self.local_slots.truncate(MAX_LOCAL_LIGHTS);
    }

    pub fn local_light_count(&self) -> usize {
        self.local_lights.len()
    }

    /// Replace the projection used by subsequent frames.
    pub fn set_projection(&mut self, projection: &TMat4<f32>) {
        self.vp.projection = *projection;
    }

    /// Apply the renderer-side settings of a scene's environment.
    pub fn apply_environment(&mut self, env: &Environment) {
        self.sun_shadows = env.sun_shadows;
        self.exposure = env.exposure;
        self.fxaa_enabled = env.fxaa;
        self.shadow_softness = env.shadow_softness;
        self.ao_radius = env.ao_radius;
        self.ao_composite_scale = env.ao_strength;
        self.set_ambient(env.ambient_color, env.ambient_intensity);
    }

    pub fn set_ambient(&mut self, color: [f32; 3], intensity: f32) {
        self.ambient_buffer = Buffer::from_data(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            ambient_frag::Ambient_Data { color, intensity },
        )
        .unwrap();
    }

    pub fn set_view(&mut self, view: &TMat4<f32>) {
        self.vp.view = view.clone();
        let look = inverse(view);
        self.vp.camera_pos = vec3(look[12], look[13], look[14]);
        self.vp_buffer = Buffer::from_data(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            deferred_vert::VP_Data {
                view: self.vp.view.into(),
                projection: self.vp.projection.into(),
            },
        )
        .unwrap();

        let vp_layout = self
            .deferred_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();
        self.vp_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, self.vp_buffer.clone())],
            [],
        )
        .unwrap();
    }

    /// The sun's shadow map covers a square `SUN_SHADOW_EXTENT` metres wide that follows
    /// the camera, so shadows never run out as the player moves.
    pub fn compute_light_space_matrix(&self, light_dir: [f32; 4]) -> TMat4<f32> {
        sun_shadow_matrix(light_dir, &self.vp.view)
    }

    pub(super) fn extract_frustum_planes(vp: &TMat4<f32>) -> [[f32; 4]; 6] {
        let row = |i: usize| -> [f32; 4] { [vp[(i, 0)], vp[(i, 1)], vp[(i, 2)], vp[(i, 3)]] };
        let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
        };
        let sub = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
        };
        let norm = |p: [f32; 4]| -> [f32; 4] {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if len > 1e-8 {
                [p[0] / len, p[1] / len, p[2] / len, p[3] / len]
            } else {
                p
            }
        };

        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        [
            norm(add(r3, r0)),
            norm(sub(r3, r0)),
            norm(add(r3, r1)),
            norm(sub(r3, r1)),
            // w + z works for both -1..1 and 0..1 depth; for the latter it only reaches a
            // little behind the near plane, which keeps a few extra draws, never fewer.
            norm(add(r3, r2)),
            norm(sub(r3, r2)),
        ]
    }
}

/// The sun's shadow matrix for a camera with this view: a square around the camera,
/// moved in whole shadow-map texels so shadow edges don't shimmer as the camera moves.
fn sun_shadow_matrix(light_dir: [f32; 4], view: &TMat4<f32>) -> TMat4<f32> {
    let direction = vec3(light_dir[0], light_dir[1], light_dir[2]).normalize();
    // look_at needs an up vector that isn't parallel to the light.
    let up = if direction.y.abs() > 0.99 { vec3(0.0, 0.0, 1.0) } else { vec3(0.0, 1.0, 0.0) };
    let rotation = look_at_rh(&vec3(0.0, 0.0, 0.0), &direction, &up);

    // Centre the square a little ahead of the camera: what's in front is what's seen,
    // but shadows from just behind still fall into view.
    let camera = inverse(view);
    let forward = -vec3(camera[8], camera[9], camera[10]);
    let mut center = vec3(camera[12], camera[13], camera[14]) + forward * (SUN_SHADOW_EXTENT * 0.25);

    // Move in whole shadow-map texels. A map that slides smoothly samples the scene at
    // different spots every frame, which makes shadow edges shimmer and crawl.
    let texel = SUN_SHADOW_EXTENT / SHADOW_MAP_SIZE as f32;
    let in_light = rotation * nalgebra_glm::vec4(center.x, center.y, center.z, 1.0);
    let snapped = nalgebra_glm::vec4(
        (in_light.x / texel).floor() * texel,
        (in_light.y / texel).floor() * texel,
        in_light.z,
        1.0,
    );
    let back = inverse(&rotation) * snapped;
    center = vec3(back.x, back.y, back.z);

    let half = SUN_SHADOW_EXTENT * 0.5;
    let eye = center - direction * SUN_SHADOW_DEPTH * 0.5;
    let light_view = look_at_rh(&eye, &center, &up);
    let light_projection = ortho(-half, half, -half, half, 0.1, SUN_SHADOW_DEPTH);

    let vulkan_depth_correction = nalgebra_glm::mat4(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
    );
    vulkan_depth_correction * light_projection * light_view
}

/// glm's projections map depth to -1..1; Vulkan wants 0..1.
fn vulkan_depth() -> TMat4<f32> {
    nalgebra_glm::mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0)
}

fn light_view(position: [f32; 3], direction: TVec3<f32>) -> TMat4<f32> {
    let eye = vec3(position[0], position[1], position[2]);
    let up = if direction.y.abs() > 0.99 { vec3(0.0, 0.0, 1.0) } else { vec3(0.0, 1.0, 0.0) };
    look_at_rh(&eye, &(eye + direction), &up)
}

/// A spot light's view and projection, covering its cone with a little margin.
fn spot_matrix(position: [f32; 3], direction: [f32; 3], outer_degrees: f32, range: f32) -> TMat4<f32> {
    let d = vec3(direction[0], direction[1], direction[2]);
    let d = if d.norm() > 1e-6 { d.normalize() } else { vec3(0.0, 1.0, 0.0) };
    let fov = (outer_degrees * 2.0 + 10.0).clamp(10.0, 170.0).to_radians();
    vulkan_depth() * nalgebra_glm::perspective(1.0, fov, 0.05, range.max(0.1)) * light_view(position, d)
}

/// Six 95° views around a point light: +X, -X, +Y, -Y, +Z, -Z. The extra 5° gives the
/// shadow filter room at the cube's edges.
fn cube_matrices(position: [f32; 3], range: f32) -> Vec<TMat4<f32>> {
    let projection = vulkan_depth() * nalgebra_glm::perspective(1.0, 95f32.to_radians(), 0.05, range.max(0.1));
    [
        vec3(1.0, 0.0, 0.0),
        vec3(-1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, -1.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.0, -1.0),
    ]
    .into_iter()
    .map(|d| projection * light_view(position, d))
    .collect()
}

#[cfg(test)]
mod sun_shadow_tests {
    use super::*;

    /// Where a world point lands in the sun's shadow map, in texels.
    fn texel_of(matrix: &TMat4<f32>, point: TVec3<f32>) -> (f32, f32) {
        let p = matrix * nalgebra_glm::vec4(point.x, point.y, point.z, 1.0);
        let size = SHADOW_MAP_SIZE as f32;
        ((p.x / p.w * 0.5 + 0.5) * size, (p.y / p.w * 0.5 + 0.5) * size)
    }

    fn view_from(position: TVec3<f32>) -> TMat4<f32> {
        look_at_rh(&position, &(position + vec3(0.3, 0.2, 1.0)), &vec3(0.0, 1.0, 0.0))
    }

    #[test]
    fn the_map_moves_in_whole_texels_so_shadows_dont_shimmer() {
        let sun = [0.4, 1.0, 0.3, 0.0];
        let point = vec3(3.0, -0.5, 7.0);
        let first = texel_of(&sun_shadow_matrix(sun, &view_from(vec3(0.0, -1.7, 0.0))), point);
        // Walk the camera in small, uneven steps, as a player would.
        for step in 1..400 {
            let t = step as f32 * 0.0137;
            let camera = vec3(t * 1.3, -1.7 + (t * 3.0).sin() * 0.2, t * 0.7);
            let now = texel_of(&sun_shadow_matrix(sun, &view_from(camera)), point);
            // The point may land on another texel, but always at the same place within one.
            let fraction = |a: f32, b: f32| ((a - b) - (a - b).round()).abs();
            assert!(
                fraction(now.0, first.0) < 0.02 && fraction(now.1, first.1) < 0.02,
                "step {step}: point drifted within its texel: {first:?} -> {now:?}"
            );
        }
    }

    #[test]
    fn the_map_follows_the_camera() {
        let sun = [0.4, 1.0, 0.3, 0.0];
        let far = vec3(500.0, -1.0, -300.0);
        let matrix = sun_shadow_matrix(sun, &view_from(far));
        let (x, y) = texel_of(&matrix, far + vec3(0.3, 0.2, 1.0) * 2.0);
        let size = SHADOW_MAP_SIZE as f32;
        assert!(x > 0.0 && x < size && y > 0.0 && y < size, "in front of a far camera is outside the map: {x}, {y}");
    }
}
