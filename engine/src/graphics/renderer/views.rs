//! Extra views of the scene, rendered into images: the editor's other viewports, camera
//! previews, and SubViewports that show on surfaces or in the UI.
//!
//! A view is its own GPU submission, recorded and finished before the main frame starts,
//! so it can reuse the frame's G-buffer and uniform buffers. The scene is rendered at the
//! main render size with the view's own projection, then scaled into the view's target;
//! since the projection uses the target's aspect ratio, nothing looks stretched.
//!
//! ```ignore
//! renderer.set_view(&view);
//! renderer.set_projection(&projection);
//! if renderer.begin_view() {
//!     app.render_scene(&mut environment); // the usual passes
//!     renderer.end_view(&target);
//! }
//! ```

use std::sync::Arc;

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, ClearColorImageInfo, CommandBufferUsage, PrimaryCommandBufferAbstract,
        SubpassEndInfo,
    },
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, sampler::Filter, view::ImageView},
    memory::allocator::AllocationCreateInfo,
    render_pass::{Framebuffer, FramebufferCreateInfo},
    sync::GpuFuture,
};

use super::{RenderStage, Renderer};

/// The format view targets use: sampled as linear colour, like model textures.
pub const VIEW_FORMAT: Format = Format::R8G8B8A8_SRGB;

impl Renderer {
    /// An image a view can be rendered into and then shown on a surface or in the UI.
    pub fn create_view_target(&self, extent: [u32; 2]) -> Arc<ImageView> {
        let image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: VIEW_FORMAT,
                extent: [extent[0].max(1), extent[1].max(1), 1],
                usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        ImageView::new_default(image).unwrap()
    }

    /// Fill a view target with one colour, for example before drawing UI into it.
    pub fn clear_view_target(&self, target: &Arc<ImageView>, color: [f32; 4]) {
        let mut commands = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        commands
            .clear_color_image(ClearColorImageInfo {
                clear_value: color.into(),
                ..ClearColorImageInfo::image(target.image().clone())
            })
            .unwrap();
        self.submit_and_wait(commands);
    }

    /// A command buffer for work outside the frame, to pass to `submit_and_wait`.
    pub fn one_time_commands(&self) -> AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }

    /// Start recording a view. Returns false (and records nothing) if a frame is being
    /// recorded already; views must be rendered before `start`.
    pub fn begin_view(&mut self) -> bool {
        if self.commands.is_some() || !matches!(self.render_stage, RenderStage::Stopped) {
            return false;
        }
        self.ensure_view_scratch();
        let commands = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        self.commands = Some(commands);
        self.render_stage = RenderStage::Deferred;
        self.view_pass = true;
        // The deferred framebuffers are all alike; the first is used for views.
        self.image_index = 0;
        true
    }

    /// Finish the view started with `begin_view`: light, post-process and tone map it as
    /// the main view would be, then scale it into `target` and wait for the GPU.
    pub fn end_view(&mut self, target: &Arc<ImageView>) {
        let finished = matches!(self.render_stage, RenderStage::Directional | RenderStage::LightObject);
        let commands = self.commands.take();
        self.render_stage = RenderStage::Stopped;
        let (Some(mut commands), true) = (commands, finished) else {
            self.view_pass = false;
            return;
        };
        commands.end_render_pass(SubpassEndInfo::default()).unwrap();
        self.dispatch_ao(&mut commands);
        self.dispatch_blur(&mut commands);
        self.dispatch_fxaa(&mut commands);
        self.dispatch_post_effects(&mut commands);
        self.composite(&mut commands);
        self.view_pass = false;

        let (scratch, _) = self.view_scratch.clone().unwrap();
        commands
            .blit_image(BlitImageInfo {
                filter: Filter::Linear,
                ..BlitImageInfo::images(scratch.image().clone(), target.image().clone())
            })
            .unwrap();
        self.submit_and_wait(commands);
    }

    /// Run recorded commands now and wait for them, outside the frame.
    pub fn submit_and_wait(&self, commands: AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>) {
        let result = commands
            .build()
            .unwrap()
            .execute(self.queue.clone())
            .map(|f| f.then_signal_fence_and_flush());
        match result {
            Ok(Ok(future)) => {
                let _ = future.wait(None);
            }
            Ok(Err(e)) => eprintln!("Failed to render a view: {e:?}"),
            Err(e) => eprintln!("Failed to render a view: {e:?}"),
        }
    }

    /// The composite target for views, at the current render size.
    fn ensure_view_scratch(&mut self) {
        let extent = self.render_extent();
        if let Some((view, _)) = &self.view_scratch {
            let [w, h, _] = view.image().extent();
            if [w, h] == extent {
                return;
            }
        }
        let image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: self.composite_format,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        let view = ImageView::new_default(image).unwrap();
        let framebuffer = Framebuffer::new(
            self.composite_render_pass.clone(),
            FramebufferCreateInfo { attachments: vec![view.clone()], ..Default::default() },
        )
        .unwrap();
        self.view_scratch = Some((view, framebuffer));
    }

    /// The perspective projection the renderer uses, for a camera's field of view (in
    /// degrees) and an image's aspect ratio.
    pub fn camera_projection(aspect: f32, fov_degrees: f32, near: f32, far: f32) -> nalgebra_glm::Mat4 {
        nalgebra_glm::perspective(aspect.max(0.01), fov_degrees.clamp(1.0, 179.0).to_radians(), near.max(0.0001), far.max(near + 0.01))
    }
}
