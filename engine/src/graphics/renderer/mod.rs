mod compute;
mod ibl;
mod images;
mod passes;
mod pipelines;
mod upload;

use std::{mem, sync::Arc};

use nalgebra_glm::{TMat4, TVec3, half_pi, identity, inverse, look_at_rh, ortho, perspective, vec3};
use sdl3::video::Window;
use vulkano::{
    Handle, Version, VulkanLibrary, VulkanObject,
    Validated, VulkanError,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo,
        allocator::StandardCommandBufferAllocator,
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
            BorderColor, Filter, LOD_CLAMP_NONE, Sampler, SamplerAddressMode,
            SamplerCreateInfo, SamplerMipmapMode,
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
use vulkano::command_buffer::{DrawIndexedIndirectCommand, PrimaryCommandBufferAbstract};

use ash::vk;

use nalgebra_glm::TVec3 as Vec3;

use crate::assets::asset_manager::{self, UnifiedGeometry};
use crate::assets::gltf_loader::DummyVertex;
use crate::scene::components::pointlight::Pointlight;
use crate::scene::components::transform::Transform;

// ── Shader modules ─────────────────────────────────────────────────────────

mod deferred_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/graphics/renderer/shaders/deferred.vert",
    }
}
mod deferred_frag {
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
mod pointlight_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/graphics/renderer/shaders/pointlight.frag",
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

use nalgebra_glm::vec3 as glm_vec3;

#[derive(Default, Debug, Clone)]
pub struct DirectionalLight {
    pub position: [f32; 4],
    pub color: [f32; 3],
}

impl DirectionalLight {
    pub fn new(position: [f32; 4], color: [f32; 3]) -> DirectionalLight {
        DirectionalLight { position, color }
    }

    pub fn get_position(&self) -> TVec3<f32> {
        glm_vec3(self.position[0], self.position[1], self.position[2])
    }
}

#[derive(Default, Debug, Clone)]
pub struct PointLight {
    pub position: [f32; 4],
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
}

impl PointLight {
    pub fn new(position: [f32; 4], color: [f32; 3], intensity: f32, radius: f32) -> PointLight {
        PointLight {
            position,
            color,
            intensity,
            radius,
        }
    }
}

// ── Internal types ─────────────────────────────────────────────────────────

const SHADOW_MAP_SIZE: u32 = 4096;

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

pub struct CulledDrawBuffers {
    pub indirect: Subbuffer<[DrawIndexedIndirectCommand]>,
    pub draw_data: Subbuffer<[asset_manager::DrawData]>,
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
    pub(super) pointlight_pipeline: Arc<GraphicsPipeline>,
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
}

// ── Core impl ─────────────────────────────────────────────────────────────

impl Renderer {
    pub fn new(window: &Window) -> Renderer {
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
            let usage = caps.supported_usage_flags;
            let alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
            let image_format = Some(
                device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );
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
                    image_usage: usage,
                    composite_alpha: alpha,
                    present_mode: PresentMode::Immediate,
                    ..Default::default()
                },
            )
            .unwrap()
        };

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
                    color: [color, normals, frag_location, specular],
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
            extent: [1600.0, 900.0],
            depth_range: 0.0..=1.0,
        };

        let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
        let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

        let composite_render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
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
        let pointlight_pipeline =
            pipelines::create_pointlight(device.clone(), viewport.clone(), lighting_pass.clone());
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
            &images,
            render_pass.clone(),
            composite_render_pass.clone(),
            swapchain.image_format(),
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
            pointlight_pipeline,
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
        self.composite(&mut commands);

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
        let clear_values = vec![
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0].into()),
            Some(1.0.into()),
        ];
        self.commands
            .as_mut()
            .unwrap()
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values,
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[self.image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();
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
            ambient_frag::Ambient_Data {
                color,
                intensity,
            },
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

    pub fn compute_light_space_matrix(light_dir: [f32; 4]) -> TMat4<f32> {
        let light_direction = vec3(light_dir[0], light_dir[1], light_dir[2]).normalize();
        let eye = vec3(0.0, 0.0, 0.0) - light_direction * 50.0;
        let target = vec3(0.0, 0.0, 0.0);
        let up = vec3(0.0, 1.0, 0.0);

        let light_view = look_at_rh(&eye, &target, &up);
        let light_projection = ortho(-25.0, 25.0, -25.0, 25.0, 0.1, 100.0);

        let vulkan_depth_correction = nalgebra_glm::mat4(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
        );

        vulkan_depth_correction * light_projection * light_view
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
            norm(r2),
            norm(sub(r3, r2)),
        ]
    }
}
