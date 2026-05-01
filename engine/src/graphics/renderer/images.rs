use std::sync::Arc;

use nalgebra_glm::{half_pi, perspective};
use sdl3::video::Window;
use vulkano::{
    Validated, VulkanError,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{Image, ImageCreateInfo, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{Pipeline, graphics::viewport::Viewport},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::SwapchainCreateInfo,
};

use super::{Renderer, deferred_vert};

impl Renderer {
    pub fn recreate_swapchain(&mut self) {
        self.render_stage = super::RenderStage::NeedsRedraw;
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

        self.render_stage = super::RenderStage::Stopped;
    }

    pub(super) fn window_size_dependent_setup(
        allocator: Arc<StandardMemoryAllocator>,
        images: &[Arc<vulkano::image::Image>],
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

        let scene_image = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: Format::R16G16B16A16_SFLOAT,
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
}
