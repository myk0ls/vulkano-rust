use std::{mem, slice, sync::Arc};

use nalgebra_glm::{TMat4, TVec3, half_pi, identity, inverse, perspective, vec3};
use sdl3::video::Window;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use vulkano::{
    Handle, Version, VulkanLibrary, VulkanObject,
    buffer::{
        BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess,
        cpu_pool::CpuBufferPoolSubbuffer,
    },
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet,
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    format::Format,
    image::{
        AttachmentImage, ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage,
        ImmutableImage, MipmapsCount, SwapchainImage, swapchain,
        view::{ImageView, ImageViewCreateInfo},
    },
    instance::{self, Instance, InstanceCreateInfo},
    library,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, StateMode,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
            },
            depth_stencil::{CompareOp, DepthBoundsState, DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::{BuffersDefinition, Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode},
    swapchain::{
        AcquireError, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo,
        SwapchainCreationError, SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
};

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, Subbuffer};

use ash::vk::{self, ImageCreateInfo, ImageUsageFlags};

use vulkano::command_buffer::{PrimaryCommandBufferAbstract, SubpassBeginInfo, SubpassEndInfo};

use vulkano::swapchain::acquire_next_image;

use vulkano_win::{VkSurfaceBuild, required_extensions};

use vulkano::instance::InstanceExtensions;

use smallvec::smallvec;

use crate::{
    engine::{
        assets::{
            self,
            gltf_loader::{ColoredVertex, DummyVertex, NormalVertex},
        },
        core::application::{Application, Game},
        graphics::{
            mesh::Mesh,
            model::Model,
            renderer::DirectionalLight,
            skybox::{Skybox, SkyboxImages},
        },
        scene::components::transform::{self, Transform},
    },
    game::my_game::MyGame,
};

vulkano::impl_vertex!(DummyVertex, position);
vulkano::impl_vertex!(NormalVertex, position, normal, color, uv);
vulkano::impl_vertex!(ColoredVertex, position, color);

mod deferred_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/engine/graphics/renderer/shaders/deferred.vert",
    }
}

mod deferred_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/engine/graphics/renderer/shaders/deferred.frag",
    }
}

mod directional_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/engine/graphics/renderer/shaders/directional.vert"
    }
}

mod directional_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/engine/graphics/renderer/shaders/directional.frag",
    }
}

mod ambient_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/engine/graphics/renderer/shaders/ambient.vert"
    }
}

mod ambient_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/engine/graphics/renderer/shaders/ambient.frag",
    }
}

mod light_obj_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/engine/graphics/renderer/shaders/light_obj.vert",
    }
}

mod light_obj_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/engine/graphics/renderer/shaders/light_obj.frag"
    }
}

mod skybox_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/engine/graphics/renderer/shaders/skybox.vert",
    }
}

mod skybox_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/engine/graphics/renderer/shaders/skybox.frag",
    }
}

#[derive(Debug, Clone)]
enum RenderStage {
    Stopped,
    Deferred,
    Ambient,
    Directional,
    LightObject,
    NeedsRedraw,
}

pub struct Renderer {
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    vp: VP,
    swapchain: Arc<Swapchain>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vp_buffer: Arc<CpuAccessibleBuffer<deferred_vert::VP_Data>>,
    //model_uniform_buffer: CpuBufferPool<deferred_vert::ty::Model_Data>,
    ambient_buffer: Arc<CpuAccessibleBuffer<ambient_frag::Ambient_Data>>,
    directional_subbuffer: Subbuffer<directional_frag::Directional_Light_Data>,
    frag_location_buffer: Arc<ImageView<AttachmentImage>>,
    specular_buffer: Arc<ImageView<AttachmentImage>>,
    sampler: Arc<Sampler>,
    render_pass: Arc<RenderPass>,
    deferred_pipeline: Arc<GraphicsPipeline>,
    directional_pipeline: Arc<GraphicsPipeline>,
    ambient_pipeline: Arc<GraphicsPipeline>,
    light_obj_pipeline: Arc<GraphicsPipeline>,
    skybox_pipeline: Arc<GraphicsPipeline>,
    dummy_verts: Arc<CpuAccessibleBuffer<[DummyVertex]>>,
    framebuffers: Vec<Arc<Framebuffer>>,
    color_buffer: Arc<ImageView<AttachmentImage>>,
    normal_buffer: Arc<ImageView<AttachmentImage>>,
    vp_set: Arc<PersistentDescriptorSet>,
    viewport: Viewport,
    render_stage: RenderStage,
    pub commands: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    image_index: u32,
    acquire_future: Option<SwapchainAcquireFuture>,
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
                    max_api_version: Some(Version::V1_1),
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
            ..DeviceExtensions::empty()
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
            vp.projection = perspective(aspect_ratio, half_pi(), 0.01, 100.0);

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: caps.min_image_count,
                    image_format: image_format.unwrap(),
                    image_extent,
                    image_usage: usage,
                    composite_alpha: alpha,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(device.clone(), Default::default());
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
                    store_op: DontCare,
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
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
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
                        viewports: [viewport].into_iter().collect(),
                        ..Default::default()
                    }),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::Front,
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

        // let _directional_pipeline = GraphicsPipeline::start()
        //     .vertex_input_state(BuffersDefinition::new().vertex::<DummyVertex>())
        //     .vertex_shader(directional_vert.entry_point("main").unwrap(), ())
        //     .input_assembly_state(InputAssemblyState::new())
        //     .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        //     .fragment_shader(directional_frag.entry_point("main").unwrap(), ())
        //     .color_blend_state(
        //         ColorBlendState::new(lighting_pass.num_color_attachments()).blend(
        //             AttachmentBlend {
        //                 color_op: BlendOp::Add,
        //                 color_source: BlendFactor::One,
        //                 color_destination: BlendFactor::One,
        //                 alpha_op: BlendOp::Max,
        //                 alpha_source: BlendFactor::One,
        //                 alpha_destination: BlendFactor::One,
        //             },
        //         ),
        //     )
        //     .render_pass(lighting_pass.clone())
        //     .build(device.clone())
        //     .unwrap();

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
                        viewports: [viewport].into_iter().collect(),
                        ..Default::default()
                    }),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        lighting_pass.num_color_attachments(),
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
                    subpass: Some(lighting_pass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        // let _ambient_pipeline = GraphicsPipeline::start()
        //     .vertex_input_state(BuffersDefinition::new().vertex::<DummyVertex>())
        //     .vertex_shader(ambient_vert.entry_point("main").unwrap(), ())
        //     .input_assembly_state(InputAssemblyState::new())
        //     .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        //     .fragment_shader(ambient_frag.entry_point("main").unwrap(), ())
        //     .color_blend_state(
        //         ColorBlendState::new(lighting_pass.num_color_attachments()).blend(
        //             AttachmentBlend {
        //                 color_op: BlendOp::Add,
        //                 color_source: BlendFactor::One,
        //                 color_destination: BlendFactor::One,
        //                 alpha_op: BlendOp::Max,
        //                 alpha_source: BlendFactor::One,
        //                 alpha_destination: BlendFactor::One,
        //             },
        //         ),
        //     )
        //     .render_pass(lighting_pass.clone())
        //     .build(device.clone())
        //     .unwrap();

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
                        viewports: [viewport].into_iter().collect(),
                        ..Default::default()
                    }),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        lighting_pass.num_color_attachments(),
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
                    subpass: Some(lighting_pass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        // let _light_obj_pipeline = GraphicsPipeline::start()
        //     .vertex_input_state(BuffersDefinition::new().vertex::<ColoredVertex>())
        //     .vertex_shader(light_obj_vert.entry_point("main").unwrap(), ())
        //     .input_assembly_state(InputAssemblyState::new())
        //     .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        //     .fragment_shader(light_obj_frag.entry_point("main").unwrap(), ())
        //     .depth_stencil_state(DepthStencilState::simple_depth_test())
        //     .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        //     .render_pass(lighting_pass.clone())
        //     .build(device.clone())
        //     .unwrap();

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
                        viewports: [viewport].into_iter().collect(),
                        ..Default::default()
                    }),
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
                        lighting_pass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    subpass: Some(lighting_pass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        // let _skybox_pipeline = GraphicsPipeline::start()
        //     .vertex_input_state(BuffersDefinition::new().vertex::<DummyVertex>())
        //     .vertex_shader(skybox_vert.entry_point("main").unwrap(), ())
        //     .input_assembly_state(InputAssemblyState::new())
        //     .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        //     .fragment_shader(skybox_frag.entry_point("main").unwrap(), ())
        //     .depth_stencil_state(DepthStencilState {
        //         depth: Some(DepthState {
        //             write_enable: StateMode::Fixed(false),
        //             compare_op: StateMode::Fixed(CompareOp::LessOrEqual),
        //             ..Default::default()
        //         }),
        //         depth_bounds: None,
        //         stencil: None,
        //     })
        //     .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
        //     .render_pass(lighting_pass.clone())
        //     .build(device.clone())
        //     .unwrap();

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
                        viewports: [viewport].into_iter().collect(),
                        ..Default::default()
                    }),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::None,
                        ..Default::default()
                    }),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState {
                            write_enable: StateMode::Fixed(false),
                            compare_op: StateMode::Fixed(CompareOp::LessOrEqual),
                        }),
                        depth_bounds: None,
                        stencil: None,
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        lighting_pass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    subpass: Some(lighting_pass.into()),
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
                ..Default::default()
            },
        );

        let directional_subbuffer: Subbuffer<directional_frag::Directional_Light_Data> =
            directional_allocator.allocate_sized().unwrap();

        let dummy_verts = Buffer::from_iter(
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
            DummyVertex::list().iter().cloned(),
        )
        .unwrap();

        let (framebuffers, color_buffer, normal_buffer, frag_location_buffer, specular_buffer) =
            Renderer::window_size_dependent_setup(
                &memory_allocator,
                &images,
                render_pass.clone(),
                &mut viewport,
            );

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Nearest,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mip_lod_bias: 0.0,
                ..Default::default()
            },
        )
        .unwrap();

        let vp_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();
        let vp_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, vp_buffer.clone())],
        )
        .unwrap();

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
            frag_location_buffer,
            specular_buffer,
            sampler,
            render_pass,
            deferred_pipeline,
            directional_pipeline,
            ambient_pipeline,
            light_obj_pipeline,
            skybox_pipeline,
            dummy_verts,
            framebuffers,
            color_buffer,
            normal_buffer,
            vp_set,
            viewport,
            render_stage,
            commands,
            image_index,
            acquire_future,
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
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain();
                    return;
                }
                Err(err) => panic!("{:?}", err),
            };

        if suboptimal {
            self.recreate_swapchain();
            return;
        }

        let clear_values = vec![
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0].into()),
            Some(1.0.into()),
        ];

        let mut commands = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        commands
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values,
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
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
            Err(FlushError::OutOfDate) => {
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

    pub fn geometry(&mut self, model: &mut assets::asset_manager::Model, transform: &Transform) {
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

        let push_constants = deferred_vert::PushConstants {
            model: transform.model_matrix().into(),
            normals: transform.normal_matrix().into(),
        };

        for mesh in model.meshes.iter() {
            self.commands
                .as_mut()
                .unwrap()
                .set_viewport(0, [self.viewport.clone()].into_iter().collect())
                .unwrap()
                .bind_pipeline_graphics(self.deferred_pipeline.clone())
                .unwrap()
                .push_constants(self.deferred_pipeline.layout().clone(), 0, push_constants)
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.deferred_pipeline.layout().clone(),
                    0,
                    (
                        self.vp_set.clone(),
                        mesh.persist_desc_set.as_ref().unwrap().clone(),
                    ),
                )
                .unwrap()
                .bind_vertex_buffers(0, mesh.vertex_buffer.as_ref().unwrap().clone())
                .unwrap()
                .bind_index_buffer(mesh.index_buffer.as_ref().unwrap().clone())
                .unwrap();

            unsafe {
                self.commands
                    .as_mut()
                    .unwrap()
                    .draw_indexed(mesh.index_buffer.as_ref().unwrap().len() as u32, 1, 0, 0, 0)
                    .unwrap();
            }
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
        let ambient_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            ambient_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.color_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.ambient_buffer.clone()),
            ],
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .next_subpass(SubpassContents::Inline)
            .unwrap()
            .bind_pipeline_graphics(self.ambient_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.ambient_pipeline.layout().clone(),
                0,
                ambient_set.clone(),
            )
            .set_viewport(0, [self.viewport.clone()])
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .draw(self.dummy_verts.len() as u32, 1, 0, 0)
            .unwrap();
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

        let camera_buffer = CpuAccessibleBuffer::from_data(
            &self.memory_allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            directional_frag::ty::Camera_Data {
                position: self.vp.camera_pos.into(),
            },
        )
        .unwrap();

        let directional_subbuffer =
            self.generate_directional_buffer(&self.directional_buffer, &directional_light);

        let directional_layout = self
            .directional_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();
        let directional_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            directional_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.color_buffer.clone()),
                WriteDescriptorSet::image_view(1, self.normal_buffer.clone()),
                WriteDescriptorSet::image_view(2, self.frag_location_buffer.clone()),
                WriteDescriptorSet::image_view(3, self.specular_buffer.clone()),
                WriteDescriptorSet::buffer(4, directional_subbuffer.clone()),
                WriteDescriptorSet::buffer(5, camera_buffer.clone()),
            ],
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .set_viewport(0, [self.viewport.clone()])
            .bind_pipeline_graphics(self.directional_pipeline.clone())
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.directional_pipeline.layout().clone(),
                0,
                directional_set.clone(),
            )
            .draw(self.dummy_verts.len() as u32, 1, 0, 0)
            .unwrap();
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

        let inv_vp_buffer = CpuAccessibleBuffer::from_data(
            &self.memory_allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            skybox_frag::ty::VP_Data {
                invProjection: self.vp.projection.try_inverse().unwrap().into(),
                invView: self.vp.view.try_inverse().unwrap().into(),
            },
        )
        .unwrap();

        let skybox_layout = self.skybox_pipeline.layout().set_layouts().get(0).unwrap();
        let skybox_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            skybox_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, inv_vp_buffer.clone()),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    skybox.cubemap.clone(),
                    self.sampler.clone(),
                ),
            ],
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .set_viewport(0, [self.viewport.clone()])
            .bind_pipeline_graphics(self.skybox_pipeline.clone())
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.skybox_pipeline.layout().clone(),
                0,
                skybox_set.clone(),
            )
            .draw(self.dummy_verts.len() as u32, 1, 0, 0)
            .unwrap();
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
        let model_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            model_layout.clone(),
            //[WriteDescriptorSet::buffer(0, model_subbuffer.clone())],
            [],
        )
        .unwrap();

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_allocator,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            //model.meshes()[0].vertices.iter().cloned(),
            model.color_data().iter().cloned(),
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .bind_pipeline_graphics(self.light_obj_pipeline.clone())
            .push_constants(self.light_obj_pipeline.layout().clone(), 0, push_constants)
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.light_obj_pipeline.layout().clone(),
                0,
                (self.vp_set.clone(), model_set.clone()),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
    }

    fn generate_directional_buffer(
        &self,
        pool: &CpuBufferPool<directional_frag::ty::Directional_Light_Data>,
        light: &DirectionalLight,
    ) -> Arc<CpuBufferPoolSubbuffer<directional_frag::ty::Directional_Light_Data>> {
        let uniform_data = directional_frag::ty::Directional_Light_Data {
            position: light.position.into(),
            color: light.color.into(),
        };

        pool.from_data(uniform_data).unwrap()
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
        self.vp.projection = perspective(aspect_ratio, half_pi(), 0.01, 100.0);

        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent,
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
        };

        let (
            new_framebuffers,
            new_color_buffer,
            new_normal_buffer,
            new_frag_location_buffer,
            new_specular_buffer,
        ) = Renderer::window_size_dependent_setup(
            &self.memory_allocator,
            &new_images,
            self.render_pass.clone(),
            &mut self.viewport,
        );

        self.swapchain = new_swapchain;
        self.framebuffers = new_framebuffers;
        self.color_buffer = new_color_buffer;
        self.normal_buffer = new_normal_buffer;
        self.frag_location_buffer = new_frag_location_buffer;
        self.specular_buffer = new_specular_buffer;

        self.vp_buffer = CpuAccessibleBuffer::from_data(
            &self.memory_allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            deferred_vert::ty::VP_Data {
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
        self.vp_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, self.vp_buffer.clone())],
        )
        .unwrap();

        self.render_stage = RenderStage::Stopped;
    }

    fn window_size_dependent_setup(
        allocator: &StandardMemoryAllocator,
        images: &[Arc<SwapchainImage>],
        render_pass: Arc<RenderPass>,
        viewport: &mut Viewport,
    ) -> (
        Vec<Arc<Framebuffer>>,
        Arc<ImageView<AttachmentImage>>,
        Arc<ImageView<AttachmentImage>>,
        Arc<ImageView<AttachmentImage>>,
        Arc<ImageView<AttachmentImage>>,
    ) {
        let dimensions = images[0].dimensions().width_height();
        viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

        let depth_buffer = ImageView::new_default(
            AttachmentImage::transient(allocator, dimensions, Format::D16_UNORM).unwrap(),
        )
        .unwrap();

        let color_buffer = ImageView::new_default(
            AttachmentImage::transient_input_attachment(
                allocator,
                dimensions,
                Format::A2B10G10R10_UNORM_PACK32,
            )
            .unwrap(),
        )
        .unwrap();

        let normal_buffer = ImageView::new_default(
            AttachmentImage::transient_input_attachment(
                allocator,
                dimensions,
                Format::R16G16B16A16_SFLOAT,
            )
            .unwrap(),
        )
        .unwrap();

        let frag_location_buffer = ImageView::new_default(
            AttachmentImage::transient_input_attachment(
                allocator,
                dimensions,
                Format::R16G16B16A16_SFLOAT,
            )
            .unwrap(),
        )
        .unwrap();

        let specular_buffer = ImageView::new_default(
            AttachmentImage::transient_input_attachment(
                allocator,
                dimensions,
                Format::R16G16_SFLOAT,
            )
            .unwrap(),
        )
        .unwrap();

        let framebuffers = images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![
                            view,
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

        (
            framebuffers,
            color_buffer.clone(),
            normal_buffer.clone(),
            frag_location_buffer.clone(),
            specular_buffer.clone(),
        )
    }

    pub fn set_view(&mut self, view: &TMat4<f32>) {
        self.vp.view = view.clone();
        let look = inverse(&view);
        self.vp.camera_pos = vec3(look[12], look[13], look[14]);
        self.vp_buffer = CpuAccessibleBuffer::from_data(
            &self.memory_allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            deferred_vert::ty::VP_Data {
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
        self.vp_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, self.vp_buffer.clone())],
        )
        .unwrap();
    }

    pub fn upload_mesh_to_gpu(&self, mesh: &mut Mesh) {
        //create a commandbuffer for upload
        let mut upload_cmd_buf = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let mut raw_pixels: Vec<u8> = vec![];
        let mut image_dimensions: ImageDimensions = ImageDimensions::Dim2d {
            width: 1,
            height: 1,
            array_layers: 1,
        };

        //if texture is present
        if mesh.material.pbr.base_color_texture.is_some() {
            let base_texture = mesh
                .material
                .pbr
                .base_color_texture
                .as_ref()
                .unwrap()
                .clone();

            raw_pixels = mesh
                .material
                .pbr
                .base_color_texture
                .as_ref()
                .unwrap()
                .as_ref()
                .clone()
                .into_raw();

            image_dimensions = ImageDimensions::Dim2d {
                width: base_texture.dimensions().0,
                height: base_texture.dimensions().1,
                array_layers: 1,
            };
        } else {
            //use the base color
            let base_color = mesh.material.pbr.base_color_factor.clone();

            // Destructure the Vector4 into its components
            let (r, g, b, a) = (base_color.x, base_color.y, base_color.z, base_color.w);

            raw_pixels = vec![
                (r.clamp(0.0, 1.0) * 255.0) as u8,
                (g.clamp(0.0, 1.0) * 255.0) as u8,
                (b.clamp(0.0, 1.0) * 255.0) as u8,
                (a.clamp(0.0, 1.0) * 255.0) as u8,
            ];
        }

        let gpu_texture = {
            let image = ImmutableImage::from_iter(
                &self.memory_allocator,
                raw_pixels.iter().cloned(),
                image_dimensions,
                MipmapsCount::One,
                Format::R8G8B8A8_SRGB,
                &mut upload_cmd_buf,
            )
            .unwrap();
            ImageView::new_default(image).unwrap()
        };

        let _upload_commands = upload_cmd_buf
            .build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        mesh.texture = Some(gpu_texture);

        //vertex,index,persistendescset

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_allocator,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            mesh.vertices.iter().cloned(),
        )
        .unwrap();

        let index_buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_allocator,
            BufferUsage {
                index_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            mesh.indices.iter().cloned(),
        )
        .unwrap();

        mesh.vertex_buffer = Some(vertex_buffer);
        mesh.index_buffer = Some(index_buffer);

        let specular_buffer = CpuAccessibleBuffer::from_data(
            &self.memory_allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            deferred_frag::ty::Specular_Data {
                intensity: 0.5,
                shininess: 32.0,
            },
        )
        .unwrap();

        let model_layout = self
            .deferred_pipeline
            .layout()
            .set_layouts()
            .get(1)
            .unwrap();
        let model_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            model_layout.clone(),
            [
                WriteDescriptorSet::buffer(1, specular_buffer.clone()),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    mesh.texture.as_ref().unwrap().clone(),
                    self.sampler.clone(),
                ),
            ],
        )
        .unwrap();

        mesh.persist_desc_set = Some(model_set);
    }

    pub fn upload_skybox(&self, skybox_images: SkyboxImages) -> Skybox {
        let mut upload_cmd_buf = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let image_dimensions = ImageDimensions::Dim2d {
            width: 512,
            height: 512,
            array_layers: 6,
        };

        let all_pixels: Vec<u8> = skybox_images.faces.into_iter().flatten().collect();

        let staging = CpuAccessibleBuffer::from_iter(
            &self.memory_allocator,
            BufferUsage {
                transfer_src: true,
                ..BufferUsage::empty()
            },
            false,
            all_pixels.clone(),
        )
        .unwrap();

        let (cubemap, init) = ImmutableImage::uninitialized(
            &self.memory_allocator,
            image_dimensions,
            Format::R8G8B8A8_SRGB,
            1,
            ImageUsage {
                sampled: true,
                transfer_dst: true,
                ..Default::default()
            },
            ImageCreateFlags {
                cube_compatible: true,
                ..ImageCreateFlags::default()
            },
            vulkano::image::ImageLayout::TransferDstOptimal,
            [self.queue.queue_family_index()],
        )
        .unwrap();

        upload_cmd_buf
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                staging.clone(),
                init.clone(),
            ))
            .unwrap();

        let cubemap_view = ImageView::new(
            cubemap.clone(),
            ImageViewCreateInfo {
                view_type: vulkano::image::ImageViewType::Cube,
                ..ImageViewCreateInfo::from_image(&cubemap)
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
}
