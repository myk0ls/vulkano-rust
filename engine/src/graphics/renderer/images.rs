use std::sync::Arc;

use nalgebra_glm::{half_pi, perspective};
use vulkano::{
    Validated, VulkanError,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
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

        let image_extent: [u32; 2] = self.window.size().into();

        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent,
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(Validated::ValidationError(_)) => return,
            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
        };

        self.swapchain = new_swapchain;
        self.swapchain_generation += 1;

        // Off-screen targets are sized by the owner of the viewport, not the window.
        if self.offscreen.is_none() {
            let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
            self.vp.projection = perspective(aspect_ratio, half_pi(), 0.01, 1000.0);
            self.rebuild_render_targets(&new_images);
        }
        self.swapchain_images = new_images;

        self.render_stage = super::RenderStage::Stopped;
    }

    /// Render the scene into an image of the given size instead of the swapchain.
    /// May be called again to resize. Not valid while a frame is being recorded.
    pub fn set_offscreen_extent(&mut self, extent: [u32; 2]) {
        let extent = [extent[0].max(1), extent[1].max(1)];
        let image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: self.composite_format,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        self.offscreen = Some(ImageView::new_default(image.clone()).unwrap());
        self.rebuild_render_targets(&[image]);
    }

    /// The image the scene is composited into, when rendering off-screen.
    pub fn offscreen_view(&self) -> Option<&Arc<ImageView>> {
        self.offscreen.as_ref()
    }

    /// Size of the images the scene is currently rendered at.
    pub fn render_extent(&self) -> [u32; 2] {
        let extent = self.color_buffer.image().extent();
        [extent[0], extent[1]]
    }

    /// Framebuffer index for this frame: off-screen rendering has a single target.
    pub(super) fn target_index(&self) -> usize {
        if self.offscreen.is_some() {
            0
        } else {
            self.image_index as usize
        }
    }

    fn rebuild_render_targets(&mut self, images: &[Arc<vulkano::image::Image>]) {
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
            images,
            self.render_pass.clone(),
            self.composite_render_pass.clone(),
            self.composite_format,
            &mut self.viewport,
        );

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

        let view = self.vp.view;
        self.set_view(&view);
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
