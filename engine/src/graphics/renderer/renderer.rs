use std::{mem, slice, sync::Arc};

use nalgebra_glm::{
    TMat4, TVec3, half_pi, identity, inverse, look_at_rh, ortho, perspective, vec3,
};
use sdl3::video::Window;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::DeviceFeatures;
use vulkano::image::view::ImageViewType;
use vulkano::image::{Image, ImageCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::{ComputePipeline, DynamicState, PipelineShaderStageCreateInfo};
use vulkano::{
    Handle, Version, VulkanLibrary, VulkanObject,
    buffer::BufferUsage,
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyBufferToImageInfo,
        ImageBlit, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        WriteDescriptorSet,
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    format::Format,
    image::sampler::{
        Filter, LOD_CLAMP_NONE, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
    },
    image::{
        ImageCreateFlags, ImageUsage,
        view::{ImageView, ImageViewCreateInfo},
    },
    instance::{self, Instance, InstanceCreateInfo},
    library,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
            },
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::{BuffersDefinition, Vertex, VertexDefinition, VertexInputState},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        PresentMode, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo,
        SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
};

use vulkano::descriptor_set::layout::DescriptorBindingFlags;
use vulkano::{Validated, VulkanError, swapchain::acquire_next_image};

use vulkano::buffer::{Buffer, BufferCreateInfo, Subbuffer};

use ash::vk;

use vulkano::command_buffer::DrawIndexedIndirectCommand;
use vulkano::command_buffer::{PrimaryCommandBufferAbstract, SubpassBeginInfo, SubpassEndInfo};

use vulkano::instance::InstanceExtensions;

use vulkano::image::sampler::BorderColor;
use vulkano::pipeline::graphics::rasterization::DepthBiasState;

use crate::assets::asset_manager::{self, UnifiedGeometry};
use crate::graphics::renderer::PointLight;
use crate::{
    assets::{
        self,
        gltf_loader::{ColoredVertex, DummyVertex, NormalVertex},
    },
    graphics::{
        mesh::Mesh,
        model::Model,
        renderer::DirectionalLight,
        skybox::{Skybox, SkyboxImages},
    },
    scene::components::transform::{self, Transform},
};

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

pub struct Renderer {
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    vp: VP,
    swapchain: Arc<Swapchain>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vp_buffer: Subbuffer<deferred_vert::VP_Data>,
    //model_uniform_buffer: CpuBufferPool<deferred_vert::ty::Model_Data>,
    ambient_buffer: Subbuffer<ambient_frag::Ambient_Data>,
    directional_subbuffer: Subbuffer<directional_frag::Directional_Light_Data>,
    directional_allocator: SubbufferAllocator,
    frag_location_buffer: Arc<ImageView>,
    specular_buffer: Arc<ImageView>,
    sampler: Arc<Sampler>,
    shadow_sampler: Arc<Sampler>,
    shadow_map_view: Arc<ImageView>,
    render_pass: Arc<RenderPass>,
    shadow_render_pass: Arc<RenderPass>,
    shadow_pipeline: Arc<GraphicsPipeline>,
    deferred_pipeline: Arc<GraphicsPipeline>,
    directional_pipeline: Arc<GraphicsPipeline>,
    pointlight_pipeline: Arc<GraphicsPipeline>,
    ambient_pipeline: Arc<GraphicsPipeline>,
    light_obj_pipeline: Arc<GraphicsPipeline>,
    skybox_pipeline: Arc<GraphicsPipeline>,
    ao_pipeline: Arc<ComputePipeline>,
    blur_pipeline: Arc<ComputePipeline>,
    fxaa_pipeline: Arc<ComputePipeline>,
    composite_pipeline: Arc<GraphicsPipeline>,
    composite_render_pass: Arc<RenderPass>,
    composite_framebuffers: Vec<Arc<Framebuffer>>,
    scene_image: Arc<ImageView>,
    dummy_verts: Subbuffer<[DummyVertex]>,
    framebuffers: Vec<Arc<Framebuffer>>,
    shadow_framebuffer: Arc<Framebuffer>,
    color_buffer: Arc<ImageView>,
    normal_buffer: Arc<ImageView>,
    depth_buffer: Arc<ImageView>,
    ao_image: Arc<ImageView>,
    ao_blurred_image: Arc<ImageView>,
    fxaa_image: Arc<ImageView>,
    ao_sampler: Arc<Sampler>,
    ao_repeat_sampler: Arc<Sampler>,
    ao_rotation_image: Arc<ImageView>,
    vp_set: Arc<DescriptorSet>,
    bindless_material_set: Option<Arc<DescriptorSet>>,
    viewport: Viewport,
    render_stage: RenderStage,
    pub commands: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    image_index: u32,
    acquire_future: Option<SwapchainAcquireFuture>,
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

impl Renderer {
    pub fn new(window: &Window) -> Renderer {
        let instance = {
            let library = VulkanLibrary::new().unwrap();
            //let extensions = vulkano_win::required_extensions(&library);
            let sdl_extensions = window.vulkan_instance_extensions().unwrap();

            // Convert Vec<String> -> InstanceExtensions safely
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

        // SDL wants *mut VkInstance, so turn that u64 into a pointer
        let raw_instance_ptr = raw_instance.as_raw() as *mut vk::Instance;

        // Now call SDL to create the surface
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

        // let surface = WindowBuilder::new()
        //     .build_vk_surface(event_loop, instance.clone())
        //     .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ext_descriptor_indexing: true,
            ..DeviceExtensions::empty()
        };

        let device_features = DeviceFeatures {
            // Required for bindless textures
            descriptor_binding_partially_bound: true,
            runtime_descriptor_array: true,
            descriptor_binding_variable_descriptor_count: true,
            sampler_anisotropy: true,
            shader_sampled_image_array_non_uniform_indexing: true,
            multi_draw_indirect: true,
            shader_draw_parameters: true,
            ..DeviceFeatures::empty()
        };

        //let device_features = BindlessContext::required_features(&instance);

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // pick first queue_familiy_index that handles graphics and can draw on the surface created by winit
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| {
                // lower score for preferred device types
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
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
                    //Fifo locks to refresh rate, Immediate mode for immediate rendering, mailbox low latency, no tear
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

        let deferred_vert = deferred_vert::load(device.clone()).unwrap();
        let deferred_frag = deferred_frag::load(device.clone()).unwrap();
        let directional_vert = directional_vert::load(device.clone()).unwrap();
        let directional_frag = directional_frag::load(device.clone()).unwrap();
        let ambient_vert = ambient_vert::load(device.clone()).unwrap();
        let ambient_frag = ambient_frag::load(device.clone()).unwrap();
        let light_obj_frag = light_obj_frag::load(device.clone()).unwrap();
        let light_obj_vert = light_obj_vert::load(device.clone()).unwrap();
        let skybox_vert = skybox_vert::load(device.clone()).unwrap();
        let skybox_frag = skybox_frag::load(device.clone()).unwrap();
        let pointlight_vert = pointlight_vert::load(device.clone()).unwrap();
        let pointlight_frag = pointlight_frag::load(device.clone()).unwrap();
        let shadows_vert = shadows_vert::load(device.clone()).unwrap();
        let shadows_frag = shadows_frag::load(device.clone()).unwrap();
        let ao_comp = ao_comp::load(device.clone()).unwrap();
        let composite_vert = composite_vert::load(device.clone()).unwrap();
        let composite_frag = composite_frag::load(device.clone()).unwrap();

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
                    format: swapchain.image_format(),
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

        let deferred_pipeline = {
            let vs = deferred_vert.entry_point("main").unwrap();
            let fs = deferred_frag.entry_point("main").unwrap();

            let vertex_input_state = NormalVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let mut layout_create_info =
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);

            // Patch set 1 (material set) for bindless textures:
            // binding 1 is the `sampler2D textures[]` array
            if let Some(set1) = layout_create_info.set_layouts.get_mut(1) {
                if let Some(binding) = set1.bindings.get_mut(&1) {
                    binding.binding_flags |= DescriptorBindingFlags::PARTIALLY_BOUND
                        | DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
                    binding.descriptor_count = 65536;
                }
            }

            let layout = PipelineLayout::new(
                device.clone(),
                layout_create_info
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [viewport.clone()].into_iter().collect(),
                        ..Default::default()
                    }),
                    //dynamic_state: [DynamicState::ViewportWithCount].into_iter().collect(),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::Back,
                        ..Default::default()
                    }),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        deferred_pass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    subpass: Some(deferred_pass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let directional_pipeline = {
            let vs = directional_vert.entry_point("main").unwrap();
            let fs = directional_frag.entry_point("main").unwrap();

            let vertex_input_state = DummyVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [viewport.clone()].into_iter().collect(),
                        ..Default::default()
                    }),
                    //dynamic_state: [DynamicState::ViewportWithCount].into_iter().collect(),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: None,
                        ..Default::default()
                    }),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        lighting_pass.clone().num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                src_color_blend_factor: BlendFactor::One,
                                dst_color_blend_factor: BlendFactor::One,
                                color_blend_op: BlendOp::Add,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::One,
                                alpha_blend_op: BlendOp::Max,
                            }),
                            ..Default::default()
                        },
                    )),
                    subpass: Some(lighting_pass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let pointlight_pipeline = {
            let vs = pointlight_vert.entry_point("main").unwrap();
            let fs = pointlight_frag.entry_point("main").unwrap();

            let vertex_input_state = DummyVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [viewport.clone()].into_iter().collect(),
                        ..Default::default()
                    }),
                    //dynamic_state: [DynamicState::ViewportWithCount].into_iter().collect(),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: None,
                        ..Default::default()
                    }),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        lighting_pass.clone().num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                src_color_blend_factor: BlendFactor::One,
                                dst_color_blend_factor: BlendFactor::One,
                                color_blend_op: BlendOp::Add,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::One,
                                alpha_blend_op: BlendOp::Max,
                            }),
                            ..Default::default()
                        },
                    )),
                    subpass: Some(lighting_pass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let ambient_pipeline = {
            let vs = ambient_vert.entry_point("main").unwrap();
            let fs = ambient_frag.entry_point("main").unwrap();

            let vertex_input_state = DummyVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [viewport.clone()].into_iter().collect(),
                        ..Default::default()
                    }),
                    //dynamic_state: [DynamicState::ViewportWithCount].into_iter().collect(),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: None,
                        ..Default::default()
                    }),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        lighting_pass.clone().num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                src_color_blend_factor: BlendFactor::One,
                                dst_color_blend_factor: BlendFactor::One,
                                color_blend_op: BlendOp::Add,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::One,
                                alpha_blend_op: BlendOp::Max,
                            }),
                            ..Default::default()
                        },
                    )),
                    subpass: Some(lighting_pass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let light_obj_pipeline = {
            let vs = light_obj_vert.entry_point("main").unwrap();
            let fs = light_obj_frag.entry_point("main").unwrap();

            let vertex_input_state = ColoredVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [viewport.clone()].into_iter().collect(),
                        ..Default::default()
                    }),
                    //dynamic_state: [DynamicState::ViewportWithCount].into_iter().collect(),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::Back,
                        ..Default::default()
                    }),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        lighting_pass.clone().num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    subpass: Some(lighting_pass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let skybox_pipeline = {
            let vs = skybox_vert.entry_point("main").unwrap();
            let fs = skybox_frag.entry_point("main").unwrap();

            let vertex_input_state = DummyVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [viewport.clone()].into_iter().collect(),
                        ..Default::default()
                    }),
                    //dynamic_state: [DynamicState::ViewportWithCount].into_iter().collect(),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::None,
                        ..Default::default()
                    }),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState {
                            write_enable: false,
                            compare_op: CompareOp::LessOrEqual,
                        }),
                        depth_bounds: None,
                        stencil: None,
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        lighting_pass.clone().num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    subpass: Some(lighting_pass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let shadow_pass = Subpass::from(shadow_render_pass.clone(), 0).unwrap();

        let shadow_pipeline = {
            let vs = shadows_vert.entry_point("main").unwrap();
            let fs = shadows_frag.entry_point("main").unwrap();

            let vertex_input_state = NormalVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [Viewport {
                            offset: [0.0, 0.0],
                            extent: [SHADOW_MAP_SIZE as f32, SHADOW_MAP_SIZE as f32],
                            depth_range: 0.0..=1.0,
                        }]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    }),
                    //dynamic_state: [DynamicState::ViewportWithCount].into_iter().collect(),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::Front,
                        depth_bias: Some(DepthBiasState {
                            constant_factor: 0.0, // fights shadow acne
                            clamp: 0.0,
                            slope_factor: 0.0, //was 1.75
                        }),
                        ..Default::default()
                    }),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    // color_blend_state: Some(ColorBlendState::with_attachment_states(
                    //     0,
                    //     Default::default(),
                    // )),
                    color_blend_state: None,
                    subpass: Some(shadow_pass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let ao_pipeline = {
            let cs = ao_comp.entry_point("main").unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let blur_comp = blur_comp::load(device.clone()).unwrap();
        let blur_pipeline = {
            let cs = blur_comp.entry_point("main").unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let fxaa_comp = fxaa_comp::load(device.clone()).unwrap();
        let fxaa_pipeline = {
            let cs = fxaa_comp.entry_point("main").unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

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

        let composite_pipeline = {
            let vs = composite_vert.entry_point("main").unwrap();
            let fs = composite_frag.entry_point("main").unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(VertexInputState::new()),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [viewport.clone()].into_iter().collect(),
                        ..Default::default()
                    }),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        1,
                        ColorBlendAttachmentState::default(),
                    )),
                    subpass: Some(composite_pass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

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

        // let model_uniform_buffer: CpuBufferPool<deferred_vert::ty::Model_Data> =
        //     CpuBufferPool::uniform_buffer(memory_allocator.clone());

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

        // let directional_buffer: CpuBufferPool<directional_frag::Directional_Light_Data> =
        //     CpuBufferPool::uniform_buffer(memory_allocator.clone());

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

        let shadow_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                mip_lod_bias: 0.0,
                compare: Some(CompareOp::LessOrEqual),
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

        // 4x4 SSAO rotation texture: 16 random vec3 values stored as RGBA8
        // Component values are in [0,1]; shader subtracts 1.0 to shift to [-1,0]
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

        let vp_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();
        let vp_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, vp_buffer.clone())],
            [],
        )
        .unwrap();

        let bindless_material_set = None;

        let render_stage = RenderStage::Stopped;

        let commands = None;
        let image_index = 0;
        let acquire_future = None;

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
            //model_uniform_buffer,
            ambient_buffer,
            directional_subbuffer,
            directional_allocator,
            frag_location_buffer,
            specular_buffer,
            sampler,
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
            bindless_material_set,
            viewport,
            render_stage,
            commands,
            image_index,
            acquire_future,
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

        let mut commands = AutoCommandBufferBuilder::primary(
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
        self.dispatch_composite(&mut commands);

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

    /// Begins the main deferred render pass.
    /// From Shadow stage to Deferred stage.
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

    pub fn shadow_pass(
        &mut self,
        light: &DirectionalLight,
        unified: &UnifiedGeometry,
        objects: &[(usize, Transform)],
    ) {
        if (objects.is_empty()) {
            self.begin_main_render_pass();
            return;
        }

        let light_space_matrix = Renderer::compute_light_space_matrix(light.position);

        // Upload light space matrix uniform
        let light_space_buffer = Buffer::from_data(
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
            shadows_vert::LightSpaceMatrix {
                lightSpaceMatrix: light_space_matrix.into(),
            },
        )
        .unwrap();

        let vb = unified.vertex_buffer.as_ref().unwrap().clone();
        let ib = unified.index_buffer.as_ref().unwrap().clone();

        let mut indirect_commands: Vec<DrawIndexedIndirectCommand> =
            Vec::with_capacity(objects.len());
        let mut draw_data_vec: Vec<asset_manager::DrawData> = Vec::with_capacity(objects.len());

        for (draw_idx, transform) in objects {
            let draw = &unified.mesh_draws[*draw_idx];

            indirect_commands.push(DrawIndexedIndirectCommand {
                index_count: draw.index_count,
                instance_count: 1,
                first_index: draw.index_offset,
                vertex_offset: draw.vertex_offset as u32,
                first_instance: 0,
            });

            draw_data_vec.push(asset_manager::DrawData {
                model: transform.model_matrix().into(),
                normals: transform.normal_matrix().into(),
                material_index: draw.material_index,
                _pad: [0; 3],
            });
        }

        let indirect_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDIRECT_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indirect_commands.into_iter(),
        )
        .unwrap();

        let draw_data_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            draw_data_vec.into_iter(),
        )
        .unwrap();

        //create descriptor sets for shadow pipelines
        // set 0 light space matrix
        let light_space_layout = self
            .shadow_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap()
            .clone();
        let light_space_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            light_space_layout,
            [WriteDescriptorSet::buffer(0, light_space_buffer)],
            [],
        )
        .unwrap();

        //set 1 draw data SSBO
        let draw_data_layout = self
            .shadow_pipeline
            .layout()
            .set_layouts()
            .get(1)
            .unwrap()
            .clone();
        let draw_data_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            draw_data_layout,
            [WriteDescriptorSet::buffer(0, draw_data_buffer)],
            [],
        )
        .unwrap();

        // Begin shadow render pass
        self.commands
            .as_mut()
            .unwrap()
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some(1.0f32.into())], // clear depth to 1.0
                    ..RenderPassBeginInfo::framebuffer(self.shadow_framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .bind_pipeline_graphics(self.shadow_pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, vb)
            .unwrap()
            .bind_index_buffer(ib)
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.shadow_pipeline.layout().clone(),
                0,
                (light_space_set, draw_data_set),
            )
            .unwrap();

        unsafe {
            self.commands
                .as_mut()
                .unwrap()
                .draw_indexed_indirect(indirect_buffer)
                .unwrap();
        }

        self.commands
            .as_mut()
            .unwrap()
            .end_render_pass(SubpassEndInfo::default())
            .unwrap();

        //now begin main render pass
        self.begin_main_render_pass();
    }

    pub fn geometry(&mut self, unified: &UnifiedGeometry, objects: &[(usize, Transform)]) {
        match self.render_stage {
            RenderStage::Deferred => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        if objects.is_empty() {
            return;
        }

        let vb = unified.vertex_buffer.as_ref().unwrap().clone();
        let ib = unified.index_buffer.as_ref().unwrap().clone();

        // Build the indirect command buffer and the per-draw data SSBO
        let mut indirect_commands: Vec<DrawIndexedIndirectCommand> =
            Vec::with_capacity(objects.len());
        let mut draw_data_vec: Vec<asset_manager::DrawData> = Vec::with_capacity(objects.len());

        for (draw_idx, transform) in objects {
            let draw = &unified.mesh_draws[*draw_idx];

            indirect_commands.push(DrawIndexedIndirectCommand {
                index_count: draw.index_count,
                instance_count: 1,
                first_index: draw.index_offset,
                vertex_offset: draw.vertex_offset as u32,
                first_instance: 0,
            });

            draw_data_vec.push(asset_manager::DrawData {
                model: transform.model_matrix().into(),
                normals: transform.normal_matrix().into(),
                material_index: draw.material_index,
                _pad: [0; 3],
            });
        }

        // Upload indirect command buffer
        let indirect_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDIRECT_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indirect_commands.into_iter(),
        )
        .unwrap();

        // Upload per-draw data SSBO
        let draw_data_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            draw_data_vec.into_iter(),
        )
        .unwrap();

        // Create descriptor set for draw data (set 2)
        let draw_data_layout = self
            .deferred_pipeline
            .layout()
            .set_layouts()
            .get(2)
            .unwrap()
            .clone();

        let draw_data_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            draw_data_layout,
            [WriteDescriptorSet::buffer(0, draw_data_buffer)],
            [],
        )
        .unwrap();

        let builder = self.commands.as_mut().unwrap();

        builder
            .bind_pipeline_graphics(self.deferred_pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, vb)
            .unwrap()
            .bind_index_buffer(ib)
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.deferred_pipeline.layout().clone(),
                0,
                (
                    self.vp_set.clone(),
                    self.bindless_material_set.as_ref().unwrap().clone(),
                    draw_data_set,
                ),
            )
            .unwrap();

        unsafe {
            builder.draw_indexed_indirect(indirect_buffer).unwrap();
        }
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
                color: color,
                intensity: intensity,
            },
        )
        .unwrap();
    }

    pub fn ambient(&mut self) {
        match self.render_stage {
            RenderStage::Deferred => {
                self.render_stage = RenderStage::Ambient;
            }
            RenderStage::Ambient => {
                return;
            }
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

        let ambient_layout = self.ambient_pipeline.layout().set_layouts().get(0).unwrap();
        let ambient_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            ambient_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.color_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.ambient_buffer.clone()),
            ],
            [],
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .next_subpass(
                SubpassEndInfo::default(),
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .bind_pipeline_graphics(self.ambient_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.ambient_pipeline.layout().clone(),
                0,
                ambient_set.clone(),
            )
            .unwrap()
            //.set_viewport(0, [self.viewport.clone()].into_iter().collect())
            //.unwrap()
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .unwrap();

        unsafe {
            self.commands
                .as_mut()
                .unwrap()
                .draw(self.dummy_verts.len() as u32, 1, 0, 0)
                .unwrap();
        }
    }

    pub fn directional(&mut self, directional_light: &DirectionalLight) {
        match self.render_stage {
            RenderStage::Ambient => {
                self.render_stage = RenderStage::Directional;
            }
            RenderStage::Directional => {}
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

        let camera_buffer = Buffer::from_data(
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
            directional_frag::Camera_Data {
                position: self.vp.camera_pos.into(),
            },
        )
        .unwrap();

        let light_space_matrix = Renderer::compute_light_space_matrix(directional_light.position);

        let light_space_buffer = Buffer::from_data(
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
            directional_frag::LightSpaceData {
                light_space_matrix: light_space_matrix.into(),
            },
        )
        .unwrap();

        let directional_subbuffer =
            self.generate_directional_buffer(&self.directional_allocator, &directional_light);

        let directional_layout = self
            .directional_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();
        let directional_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            directional_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.color_buffer.clone()),
                WriteDescriptorSet::image_view(1, self.normal_buffer.clone()),
                WriteDescriptorSet::image_view(2, self.frag_location_buffer.clone()),
                WriteDescriptorSet::image_view(3, self.specular_buffer.clone()),
                WriteDescriptorSet::buffer(4, directional_subbuffer.clone()),
                WriteDescriptorSet::buffer(5, camera_buffer.clone()),
                WriteDescriptorSet::buffer(6, light_space_buffer.clone()),
                WriteDescriptorSet::image_view_sampler(
                    7,
                    self.shadow_map_view.clone(),
                    self.shadow_sampler.clone(),
                ),
            ],
            [],
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .bind_pipeline_graphics(self.directional_pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.directional_pipeline.layout().clone(),
                0,
                directional_set.clone(),
            )
            .unwrap()
            .push_constants(
                self.directional_pipeline.layout().clone(),
                0,
                directional_frag::PushConstants {
                    shadowRadius: self.shadow_softness,
                },
            )
            .unwrap();

        unsafe {
            self.commands
                .as_mut()
                .unwrap()
                .draw(self.dummy_verts.len() as u32, 1, 0, 0)
                .unwrap();
        }
    }

    pub fn pointlight(&mut self, light: &PointLight) {
        match self.render_stage {
            RenderStage::Ambient => {
                self.render_stage = RenderStage::Directional;
            }
            RenderStage::Directional => {}
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

        let camera_buffer = Buffer::from_data(
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
            pointlight_frag::Camera_Data {
                position: self.vp.camera_pos.into(),
            },
        )
        .unwrap();

        // let directional_subbuffer =
        //     self.generate_directional_buffer(&self.directional_allocator, &directional_light);

        let point_buffer = Buffer::from_data(
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
            pointlight_frag::PointLight_Data {
                position: light.position.into(),
                color: light.color.into(),
                intensity: light.intensity.into(),
                radius: light.radius.into(),
            },
        )
        .unwrap();

        let pointlight_layout = self
            .pointlight_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();
        let pointlight_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            pointlight_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.color_buffer.clone()),
                WriteDescriptorSet::image_view(1, self.normal_buffer.clone()),
                WriteDescriptorSet::image_view(2, self.frag_location_buffer.clone()),
                WriteDescriptorSet::image_view(3, self.specular_buffer.clone()),
                WriteDescriptorSet::buffer(4, point_buffer.clone()),
                WriteDescriptorSet::buffer(5, camera_buffer.clone()),
            ],
            [],
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            //.set_viewport(0, [self.viewport.clone()].into_iter().collect())
            //.unwrap()
            .bind_pipeline_graphics(self.pointlight_pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pointlight_pipeline.layout().clone(),
                0,
                pointlight_set.clone(),
            )
            .unwrap();

        unsafe {
            self.commands
                .as_mut()
                .unwrap()
                .draw(self.dummy_verts.len() as u32, 1, 0, 0)
                .unwrap();
        }
    }

    pub fn skybox(&mut self, skybox: &mut Skybox) {
        match self.render_stage {
            RenderStage::Ambient => {
                self.render_stage = RenderStage::Directional;
            }
            RenderStage::Directional => {}
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
        };

        let inv_vp_buffer = Buffer::from_data(
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
            skybox_frag::VP_Data {
                invProjection: self.vp.projection.try_inverse().unwrap().into(),
                invView: self.vp.view.try_inverse().unwrap().into(),
            },
        )
        .unwrap();

        let skybox_layout = self.skybox_pipeline.layout().set_layouts().get(0).unwrap();
        let skybox_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            skybox_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, inv_vp_buffer.clone()),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    skybox.cubemap.clone(),
                    self.sampler.clone(),
                ),
            ],
            [],
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            //.set_viewport(0, [self.viewport.clone()].into_iter().collect())
            //.unwrap()
            .bind_pipeline_graphics(self.skybox_pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.skybox_pipeline.layout().clone(),
                0,
                skybox_set.clone(),
            )
            .unwrap();

        unsafe {
            self.commands
                .as_mut()
                .unwrap()
                .draw(self.dummy_verts.len() as u32, 1, 0, 0)
                .unwrap();
        }
    }

    pub fn light_object(&mut self, directional_light: &DirectionalLight) {
        match self.render_stage {
            RenderStage::Directional => {
                self.render_stage = RenderStage::LightObject;
            }
            RenderStage::LightObject => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        let mut model = Model::new("data/models/sphere.glb")
            .color(directional_light.color)
            .uniform_scale_factor(0.2)
            .build();

        model.translate(directional_light.get_position());

        // let model_subbuffer = {
        //     let (model_mat, normal_mat) = model.model_matrices();

        //     let uniform_data = deferred_vert::ty::Model_Data {
        //         model: model_mat.into(),
        //         normals: normal_mat.into(),
        //     };

        //     self.model_uniform_buffer.from_data(uniform_data).unwrap()
        // };
        let (model_mat, normal_mat) = model.model_matrices();
        let push_constants = light_obj_vert::PushConstants {
            model: model_mat.into(),
            normals: normal_mat.into(),
        };

        let model_layout = self
            .light_obj_pipeline
            .layout()
            .set_layouts()
            .get(1)
            .unwrap();
        let model_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            model_layout.clone(),
            //[WriteDescriptorSet::buffer(0, model_subbuffer.clone())],
            [],
            [],
        )
        .unwrap();

        // let vertex_buffer = CpuAccessibleBuffer::from_iter(
        //     &self.memory_allocator,
        //     BufferUsage {
        //         vertex_buffer: true,
        //         ..BufferUsage::empty()
        //     },
        //     false,
        //     //model.meshes()[0].vertices.iter().cloned(),
        //     model.color_data().iter().cloned(),
        // )
        // .unwrap();

        // self.commands
        //     .as_mut()
        //     .unwrap()
        //     .bind_pipeline_graphics(self.light_obj_pipeline.clone())
        //     .push_constants(self.light_obj_pipeline.layout().clone(), 0, push_constants)
        //     .bind_descriptor_sets(
        //         PipelineBindPoint::Graphics,
        //         self.light_obj_pipeline.layout().clone(),
        //         0,
        //         (self.vp_set.clone(), model_set.clone()),
        //     )
        //     .bind_vertex_buffers(0, vertex_buffer.clone())
        //     .draw(vertex_buffer.len() as u32, 1, 0, 0)
        //     .unwrap();
    }

    fn generate_directional_buffer(
        &self,
        allocator: &SubbufferAllocator,
        light: &DirectionalLight,
    ) -> Subbuffer<directional_frag::Directional_Light_Data> {
        let uniform_data = directional_frag::Directional_Light_Data {
            position: light.position.into(),
            color: light.color.into(),
        };

        let subbuffer: Subbuffer<directional_frag::Directional_Light_Data> =
            allocator.allocate_sized().unwrap();
        *subbuffer.write().unwrap() = uniform_data;
        subbuffer
    }

    pub fn recreate_swapchain(&mut self) {
        self.render_stage = RenderStage::NeedsRedraw;
        self.commands = None;

        let window = self
            .surface
            .object()
            .unwrap()
            .downcast_ref::<Window>()
            .unwrap();
        let image_extent: [u32; 2] = window.size().into();

        let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
        self.vp.projection = perspective(aspect_ratio, half_pi(), 0.01, 1000.0);

        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent,
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(Validated::ValidationError(_)) => return,
            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
        };

        let (
            new_framebuffers,
            new_composite_framebuffers,
            new_scene_image,
            new_color_buffer,
            new_normal_buffer,
            new_frag_location_buffer,
            new_specular_buffer,
            new_depth_buffer,
            new_ao_image,
            new_ao_blurred_image,
            new_fxaa_image,
        ) = Renderer::window_size_dependent_setup(
            self.memory_allocator.clone(),
            &new_images,
            self.render_pass.clone(),
            self.composite_render_pass.clone(),
            self.swapchain.image_format(),
            &mut self.viewport,
        );

        self.swapchain = new_swapchain;
        self.framebuffers = new_framebuffers;
        self.composite_framebuffers = new_composite_framebuffers;
        self.scene_image = new_scene_image;
        self.color_buffer = new_color_buffer;
        self.normal_buffer = new_normal_buffer;
        self.frag_location_buffer = new_frag_location_buffer;
        self.specular_buffer = new_specular_buffer;
        self.depth_buffer = new_depth_buffer;
        self.ao_image = new_ao_image;
        self.ao_blurred_image = new_ao_blurred_image;
        self.fxaa_image = new_fxaa_image;

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

        self.render_stage = RenderStage::Stopped;
    }

    fn window_size_dependent_setup(
        allocator: Arc<StandardMemoryAllocator>,
        images: &[Arc<Image>],
        render_pass: Arc<RenderPass>,
        composite_render_pass: Arc<RenderPass>,
        swapchain_format: Format,
        viewport: &mut Viewport,
    ) -> (
        Vec<Arc<Framebuffer>>,
        Vec<Arc<Framebuffer>>,
        Arc<ImageView>,
        Arc<ImageView>,
        Arc<ImageView>,
        Arc<ImageView>,
        Arc<ImageView>,
        Arc<ImageView>,
        Arc<ImageView>,
        Arc<ImageView>,
        Arc<ImageView>,
    ) {
        let dimensions = images[0].extent();
        viewport.extent = [dimensions[0] as f32, dimensions[1] as f32];

        // Create depth buffer
        let depth_buffer = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: Format::D32_SFLOAT,
                    extent: [dimensions[0], dimensions[1], 1],
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();

        // Create color buffer (G-buffer)
        let color_buffer = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: Format::A2B10G10R10_UNORM_PACK32,
                    extent: [dimensions[0], dimensions[1], 1],
                    usage: ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::INPUT_ATTACHMENT
                        | ImageUsage::TRANSIENT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();

        // Create normal buffer
        let normal_buffer = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: Format::R16G16B16A16_SFLOAT,
                    extent: [dimensions[0], dimensions[1], 1],
                    usage: ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::INPUT_ATTACHMENT
                        | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();

        // Create fragment location buffer
        let frag_location_buffer = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: Format::R16G16B16A16_SFLOAT,
                    extent: [dimensions[0], dimensions[1], 1],
                    usage: ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::INPUT_ATTACHMENT
                        | ImageUsage::TRANSIENT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();

        // Create specular buffer
        let specular_buffer = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: Format::R16G16_SFLOAT,
                    extent: [dimensions[0], dimensions[1], 1],
                    usage: ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::INPUT_ATTACHMENT
                        | ImageUsage::TRANSIENT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();

        let ao_image = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: Format::R8_UNORM,
                    extent: [dimensions[0], dimensions[1], 1],
                    usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();

        let ao_blurred_image = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: Format::R8_UNORM,
                    extent: [dimensions[0], dimensions[1], 1],
                    usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();

        // FXAA output image — same dimensions as scene, R8G8B8A8_UNORM supports STORAGE
        let fxaa_image = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent: [dimensions[0], dimensions[1], 1],
                    usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();

        // Intermediate scene image — lighting writes here instead of swapchain
        let scene_image = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: swapchain_format,
                    extent: [dimensions[0], dimensions[1], 1],
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();

        // Main render pass framebuffers — use scene_image as final_color target
        let framebuffers = (0..images.len())
            .map(|_| {
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![
                            scene_image.clone(),
                            color_buffer.clone(),
                            normal_buffer.clone(),
                            frag_location_buffer.clone(),
                            specular_buffer.clone(),
                            depth_buffer.clone(),
                        ],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        // Composite framebuffers — one per swapchain image
        let composite_framebuffers = images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    composite_render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        (
            framebuffers,
            composite_framebuffers,
            scene_image.clone(),
            color_buffer.clone(),
            normal_buffer.clone(),
            frag_location_buffer.clone(),
            specular_buffer.clone(),
            depth_buffer.clone(),
            ao_image.clone(),
            ao_blurred_image.clone(),
            fxaa_image.clone(),
        )
    }

    pub fn set_view(&mut self, view: &TMat4<f32>) {
        self.vp.view = view.clone();
        let look = inverse(&view);
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

    pub fn upload_texture_to_gpu(&self, mesh: &mut Mesh) {
        //create a commandbuffer for upload
        let mut upload_cmd_buf = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Collect raw pixel data
        let mut raw_pixels: Vec<u8>;
        let mut extent: [u32; 3] = [1, 1, 1];
        let mut array_layers: u32 = 1;

        if let Some(base_texture) = mesh.material.pbr.base_color_texture.as_ref() {
            raw_pixels = base_texture.as_ref().clone().into_raw();
            extent = [base_texture.dimensions().0, base_texture.dimensions().1, 1];
        } else {
            let base_color = mesh.material.pbr.base_color_factor;
            let (r, g, b, a) = (base_color.x, base_color.y, base_color.z, base_color.w);

            raw_pixels = vec![
                (r.clamp(0.0, 1.0) * 255.0) as u8,
                (g.clamp(0.0, 1.0) * 255.0) as u8,
                (b.clamp(0.0, 1.0) * 255.0) as u8,
                (a.clamp(0.0, 1.0) * 255.0) as u8,
            ];
        }

        let mip_levels = (extent[0].max(extent[1]) as f32).log2().floor() as u32 + 1;

        // --- Upload staging buffer ---
        let format = Format::R8G8B8A8_SRGB;

        let upload_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            raw_pixels.clone(), // directly upload pixels
        )
        .unwrap();

        // --- GPU Image ---
        let image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format,
                extent,
                array_layers,
                mip_levels: mip_levels,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        // --- Copy buffer to mip level 0 ---
        upload_cmd_buf
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                upload_buffer,
                image.clone(),
            ))
            .unwrap();

        // --- Generate mip chain via blit ---
        for dst_mip in 1..mip_levels {
            let src_mip = dst_mip - 1;
            let src_w = (extent[0] >> src_mip).max(1);
            let src_h = (extent[1] >> src_mip).max(1);
            let dst_w = (extent[0] >> dst_mip).max(1);
            let dst_h = (extent[1] >> dst_mip).max(1);

            upload_cmd_buf
                .blit_image(BlitImageInfo {
                    filter: Filter::Linear,
                    regions: smallvec::smallvec![ImageBlit {
                        src_subresource: vulkano::image::ImageSubresourceLayers {
                            aspects: vulkano::image::ImageAspects::COLOR,
                            mip_level: src_mip,
                            array_layers: 0..array_layers,
                        },
                        src_offsets: [[0, 0, 0], [src_w, src_h, 1]],
                        dst_subresource: vulkano::image::ImageSubresourceLayers {
                            aspects: vulkano::image::ImageAspects::COLOR,
                            mip_level: dst_mip,
                            array_layers: 0..array_layers,
                        },
                        dst_offsets: [[0, 0, 0], [dst_w, dst_h, 1]],
                        ..Default::default()
                    }],
                    ..BlitImageInfo::images(image.clone(), image.clone())
                })
                .unwrap();
        }

        let _upload_commands = upload_cmd_buf
            .build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let gpu_texture = ImageView::new_default(image).unwrap();
        mesh.texture = Some(gpu_texture);

        // --- Normal map (R8G8_UNORM: XY only, Z reconstructed in shader) ---
        if let Some(normal_map) = &mesh.material.normal {
            let img = normal_map.texture.as_ref();
            // Extract only R (X) and G (Y); shader reconstructs Z = sqrt(1 - x² - y²).
            // Saves 50% vs R8G8B8A8_UNORM.
            let raw = img.as_raw(); // [R,G,B, R,G,B, ...]
            let pixels: Vec<u8> = raw.chunks(3).flat_map(|c| [c[0], c[1]]).collect();
            mesh.normal_texture = Some(self.upload_raw_to_gpu(
                pixels,
                img.width(),
                img.height(),
                Format::R8G8_UNORM,
            ));
        }

        // --- Metallic-roughness map (R8G8_UNORM: R=metallic, G=roughness) ---
        let has_metallic = mesh.material.pbr.metallic_texture.is_some();
        let has_roughness = mesh.material.pbr.roughness_texture.is_some();
        if has_metallic || has_roughness {
            let (width, height) = if let Some(t) = &mesh.material.pbr.metallic_texture {
                (t.width(), t.height())
            } else if let Some(t) = &mesh.material.pbr.roughness_texture {
                (t.width(), t.height())
            } else {
                unreachable!()
            };
            let pixel_count = (width * height) as usize;
            let m_raw = mesh.material.pbr.metallic_texture.as_ref()
                .map(|t| t.as_raw().clone())
                .unwrap_or_else(|| vec![255u8; pixel_count]);
            let r_raw = mesh.material.pbr.roughness_texture.as_ref()
                .map(|t| t.as_raw().clone())
                .unwrap_or_else(|| vec![255u8; pixel_count]);
            // Pack into RG: 2 bytes/pixel instead of 4.
            let pixels: Vec<u8> = m_raw.iter().zip(r_raw.iter())
                .flat_map(|(&m, &r)| [m, r])
                .collect();
            mesh.mr_texture = Some(self.upload_raw_to_gpu(
                pixels,
                width,
                height,
                Format::R8G8_UNORM,
            ));
        }
    }

    /// Upload raw pixel data as a mipmapped 2D texture and return an ImageView.
    fn upload_raw_to_gpu(
        &self,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        format: Format,
    ) -> Arc<ImageView> {
        let mut cmd = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let extent = [width, height, 1];
        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

        let staging = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            pixels.into_iter(),
        )
        .unwrap();

        let image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format,
                extent,
                mip_levels,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        cmd.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, image.clone()))
            .unwrap();

        for dst_mip in 1..mip_levels {
            let src_mip = dst_mip - 1;
            let src_w = (width >> src_mip).max(1);
            let src_h = (height >> src_mip).max(1);
            let dst_w = (width >> dst_mip).max(1);
            let dst_h = (height >> dst_mip).max(1);
            cmd.blit_image(BlitImageInfo {
                filter: Filter::Linear,
                regions: smallvec::smallvec![ImageBlit {
                    src_subresource: vulkano::image::ImageSubresourceLayers {
                        aspects: vulkano::image::ImageAspects::COLOR,
                        mip_level: src_mip,
                        array_layers: 0..1,
                    },
                    src_offsets: [[0, 0, 0], [src_w, src_h, 1]],
                    dst_subresource: vulkano::image::ImageSubresourceLayers {
                        aspects: vulkano::image::ImageAspects::COLOR,
                        mip_level: dst_mip,
                        array_layers: 0..1,
                    },
                    dst_offsets: [[0, 0, 0], [dst_w, dst_h, 1]],
                    ..Default::default()
                }],
                ..BlitImageInfo::images(image.clone(), image.clone())
            })
            .unwrap();
        }

        cmd.build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        ImageView::new_default(image).unwrap()
    }

    pub fn upload_skybox(&self, skybox_images: SkyboxImages) -> Skybox {
        let mut upload_cmd_buf = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let all_pixels: Vec<u8> = skybox_images.faces.into_iter().flatten().collect();

        // let staging = CpuAccessibleBuffer::from_iter(
        //     &self.memory_allocator,
        //     BufferUsage {
        //         transfer_src: true,
        //         ..BufferUsage::empty()
        //     },
        //     false,
        //     all_pixels.clone(),
        // )
        // .unwrap();

        let staging = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            all_pixels.clone(),
        )
        .unwrap();

        // let (cubemap, init) = ImmutableImage::uninitialized(
        //     &self.memory_allocator,
        //     image_dimensions,
        //     Format::R8G8B8A8_SRGB,
        //     1,
        //     ImageUsage {
        //         sampled: true,
        //         transfer_dst: true,
        //         ..Default::default()
        //     },
        //     ImageCreateFlags {
        //         cube_compatible: true,
        //         ..ImageCreateFlags::default()
        //     },
        //     vulkano::image::ImageLayout::TransferDstOptimal,
        //     [self.queue.queue_family_index()],
        // )
        // .unwrap();

        let cube = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent: [512, 512, 1],
                array_layers: 6,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                flags: ImageCreateFlags::CUBE_COMPATIBLE,
                //initial_layout: vulkano::image::ImageLayout::TransferDstOptimal,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        upload_cmd_buf
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                staging.clone(),
                cube.clone(),
            ))
            .unwrap();

        let cubemap_view = ImageView::new(
            cube.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Cube,
                ..ImageViewCreateInfo::from_image(&cube)
            },
        )
        .unwrap();

        let _comms = upload_cmd_buf
            .build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        Skybox {
            cubemap: cubemap_view,
        }
    }

    pub fn compute_light_space_matrix(light_dir: [f32; 4]) -> TMat4<f32> {
        let light_direction = vec3(light_dir[0], light_dir[1], light_dir[2]).normalize();
        let eye = vec3(0.0, 0.0, 0.0) - light_direction * 50.0; // pull back along light dir
        let target = vec3(0.0, 0.0, 0.0);
        let up = vec3(0.0, 1.0, 0.0);

        let light_view = look_at_rh(&eye, &target, &up);
        // nalgebra_glm::ortho maps Z to [-1, 1] (OpenGL convention).
        // Vulkan expects [0, 1]. Apply a correction matrix that remaps Z:
        //   z_vulkan = z_gl * 0.5 + 0.5
        let light_projection = ortho(-25.0, 25.0, -25.0, 25.0, 0.1, 100.0);

        // Correction matrix: scales Z by 0.5 and biases by 0.5
        let vulkan_depth_correction = nalgebra_glm::mat4(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
        );

        vulkan_depth_correction * light_projection * light_view
    }

    pub fn build_bindless_material_set(&mut self, unified: &UnifiedGeometry) {
        let material_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            unified.material_data.iter().copied(),
        )
        .unwrap();

        let texture_writes: Vec<WriteDescriptorSet> = vec![
            WriteDescriptorSet::buffer(0, material_buffer),
            WriteDescriptorSet::image_view_sampler_array(
                1,
                0,
                unified
                    .textures
                    .iter()
                    .map(|iv| (iv.clone() as _, self.sampler.clone()))
                    .collect::<Vec<_>>(),
            ),
        ];

        let layout = self
            .deferred_pipeline
            .layout()
            .set_layouts()
            .get(1)
            .unwrap()
            .clone();

        let set = DescriptorSet::new_variable(
            self.descriptor_set_allocator.clone(),
            layout,
            unified.textures.len() as u32,
            texture_writes,
            [],
        )
        .unwrap();

        self.bindless_material_set = Some(set);
    }

    pub fn dispatch_ao(
        &mut self,
        commands: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let dimensions = self.swapchain.image_extent();

        let layout = self.ao_pipeline.layout().set_layouts().get(0).unwrap();
        let ao_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    self.depth_buffer.clone(),
                    self.ao_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    self.ao_rotation_image.clone(),
                    self.ao_repeat_sampler.clone(),
                ),
                WriteDescriptorSet::image_view(2, self.ao_image.clone()),
            ],
            [],
        )
        .unwrap();

        let pc = ao_comp::PushConstants {
            zNear: 0.01,
            zFar: 1000.0,
            radius: self.ao_radius,
            attScale: self.ao_att_scale,
            distScale: self.ao_dist_scale,
        };

        commands
            .bind_pipeline_compute(self.ao_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.ao_pipeline.layout().clone(),
                0,
                ao_set,
            )
            .unwrap()
            .push_constants(self.ao_pipeline.layout().clone(), 0, pc)
            .unwrap();
        unsafe {
            commands
                .dispatch([(dimensions[0] + 15) / 16, (dimensions[1] + 15) / 16, 1])
                .unwrap();
        }
    }

    pub fn dispatch_blur(
        &mut self,
        commands: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let dimensions = self.swapchain.image_extent();
        let layout = self.blur_pipeline.layout().set_layouts().get(0).unwrap();
        let groups = [(dimensions[0] + 15) / 16, (dimensions[1] + 15) / 16, 1];

        commands
            .bind_pipeline_compute(self.blur_pipeline.clone())
            .unwrap();

        // Pass 1: horizontal — ao_image -> ao_blurred_image
        let h_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    self.ao_image.clone(),
                    self.ao_sampler.clone(),
                ),
                WriteDescriptorSet::image_view(1, self.ao_blurred_image.clone()),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    self.depth_buffer.clone(),
                    self.ao_sampler.clone(),
                ),
            ],
            [],
        )
        .unwrap();

        commands
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.blur_pipeline.layout().clone(),
                0,
                h_set,
            )
            .unwrap()
            .push_constants(
                self.blur_pipeline.layout().clone(),
                0,
                blur_comp::PushConstants {
                    isHorizontal: 1,
                    depthThreshold: self.ao_blur_depth_threshold,
                },
            )
            .unwrap();
        unsafe {
            commands.dispatch(groups).unwrap();
        }

        // Pass 2: vertical — ao_blurred_image -> ao_image
        let v_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    self.ao_blurred_image.clone(),
                    self.ao_sampler.clone(),
                ),
                WriteDescriptorSet::image_view(1, self.ao_image.clone()),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    self.depth_buffer.clone(),
                    self.ao_sampler.clone(),
                ),
            ],
            [],
        )
        .unwrap();

        commands
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.blur_pipeline.layout().clone(),
                0,
                v_set,
            )
            .unwrap()
            .push_constants(
                self.blur_pipeline.layout().clone(),
                0,
                blur_comp::PushConstants {
                    isHorizontal: 0,
                    depthThreshold: self.ao_blur_depth_threshold,
                },
            )
            .unwrap();
        unsafe {
            commands.dispatch(groups).unwrap();
        }
    }

    fn dispatch_fxaa(&self, commands: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        let dimensions = self.swapchain.image_extent();
        let layout = self.fxaa_pipeline.layout().set_layouts().get(0).unwrap();
        let fxaa_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    self.scene_image.clone(),
                    self.ao_sampler.clone(),
                ),
                WriteDescriptorSet::image_view(1, self.fxaa_image.clone()),
            ],
            [],
        )
        .unwrap();

        commands
            .bind_pipeline_compute(self.fxaa_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.fxaa_pipeline.layout().clone(),
                0,
                fxaa_set,
            )
            .unwrap()
            .push_constants(
                self.fxaa_pipeline.layout().clone(),
                0,
                fxaa_comp::PushConstants {
                    enabled: self.fxaa_enabled as u32,
                    spanMax: 8.0,
                    reduceMul: 0.125,
                    reduceMin: 0.0078125,
                },
            )
            .unwrap();
        unsafe {
            commands
                .dispatch([(dimensions[0] + 15) / 16, (dimensions[1] + 15) / 16, 1])
                .unwrap();
        }
    }

    fn dispatch_composite(
        &self,
        commands: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let layout = self
            .composite_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();
        let composite_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    self.fxaa_image.clone(),
                    self.ao_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    self.ao_image.clone(),
                    self.ao_sampler.clone(),
                ),
            ],
            [],
        )
        .unwrap();

        commands
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None],
                    ..RenderPassBeginInfo::framebuffer(
                        self.composite_framebuffers[self.image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .bind_pipeline_graphics(self.composite_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.composite_pipeline.layout().clone(),
                0,
                composite_set,
            )
            .unwrap()
            .push_constants(
                self.composite_pipeline.layout().clone(),
                0,
                composite_frag::PushConstants {
                    scale: self.ao_composite_scale,
                    bias: self.ao_composite_bias,
                    exposure: self.exposure,
                },
            )
            .unwrap();
        unsafe {
            commands.draw(3, 1, 0, 0).unwrap();
        }
        commands.end_render_pass(SubpassEndInfo::default()).unwrap();
    }
}
