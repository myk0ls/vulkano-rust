//! Draws egui's output with vulkano, as the last pass of each frame.

use std::collections::HashMap;
use std::sync::Arc;

use egui::epaint::{ImageDelta, Primitive};
use egui::{ClippedPrimitive, ImageData, TextureId, TexturesDelta};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BufferImageCopy, CopyBufferToImageInfo, CopyImageToBufferInfo,
    PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    SubpassEndInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use crate::graphics::renderer::Renderer;

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450
            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec2 uv;
            layout(location = 2) in vec4 color;
            layout(location = 0) out vec2 v_uv;
            layout(location = 1) out vec4 v_color;
            layout(push_constant) uniform PushConstants {
                vec2 screen_size;
                uint texture_is_linear;
                uint target_is_srgb;
            } pc;
            void main() {
                gl_Position = vec4(2.0 * pos / pc.screen_size - 1.0, 0.0, 1.0);
                v_uv = uv;
                v_color = color;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450
            layout(location = 0) in vec2 v_uv;
            layout(location = 1) in vec4 v_color;
            layout(location = 0) out vec4 out_color;
            layout(set = 0, binding = 0) uniform sampler2D tex;
            layout(push_constant) uniform PushConstants {
                vec2 screen_size;
                uint texture_is_linear;
                uint target_is_srgb;
            } pc;

            vec3 linear_to_srgb(vec3 c) {
                return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
            }
            vec3 srgb_to_linear(vec3 c) {
                return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
            }

            void main() {
                vec4 sampled = texture(tex, v_uv);
                // egui works in gamma space, while sRGB-format images sample as linear.
                if (pc.texture_is_linear != 0) {
                    sampled.rgb = linear_to_srgb(sampled.rgb);
                }
                vec4 color = v_color * sampled;
                if (pc.target_is_srgb != 0) {
                    color.rgb = srgb_to_linear(color.rgb);
                }
                out_color = color;
            }
        ",
    }
}

#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct UiVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    color: [u8; 4],
}

struct UiTexture {
    view: Arc<ImageView>,
    set: Arc<DescriptorSet>,
    is_linear: bool,
}

pub struct Painter {
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    sampler: Arc<Sampler>,
    framebuffers: Vec<Arc<Framebuffer>>,
    framebuffer_generation: Option<u64>,
    allocator: Arc<vulkano::memory::allocator::StandardMemoryAllocator>,
    descriptors: Arc<vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator>,
    buffers: SubbufferAllocator,
    textures: HashMap<TextureId, UiTexture>,
    pending_deltas: Vec<TexturesDelta>,
    target_is_srgb: bool,
    clear: bool,
    extent: [u32; 2],
    /// Screenshot requested for the next presented frame, and one awaiting readback.
    capture_request: Option<String>,
    capture_pending: Option<(String, Subbuffer<[u8]>, [u32; 2], Format)>,
}

impl Painter {
    /// A painter that draws the UI over whatever the frame already shows, as a game does.
    pub fn new(renderer: &Renderer) -> Self {
        Self::with_clear(renderer, false)
    }

    /// With `clear`, the UI pass starts from a blank image. The editor uses this because
    /// its UI covers the whole window; a game must not, or it erases the scene.
    pub fn with_clear(renderer: &Renderer, clear: bool) -> Self {
        let device = renderer.device.clone();
        let format = renderer.swapchain_format();

        let render_pass = if clear {
            vulkano::single_pass_renderpass!(
                device.clone(),
                attachments: {
                    color: { format: format, samples: 1, load_op: Clear, store_op: Store },
                },
                pass: { color: [color], depth_stencil: {} }
            )
        } else {
            vulkano::single_pass_renderpass!(
                device.clone(),
                attachments: {
                    color: { format: format, samples: 1, load_op: Load, store_op: Store },
                },
                pass: { color: [color], depth_stencil: {} }
            )
        }
        .unwrap();

        let vs = vs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let fs = fs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let vertex_input_state = UiVertex::per_vertex().definition(&vs).unwrap();
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
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState {
                        // egui's vertices and textures are premultiplied.
                        blend: Some(AttachmentBlend {
                            src_color_blend_factor: BlendFactor::One,
                            dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                            color_blend_op: BlendOp::Add,
                            src_alpha_blend_factor: BlendFactor::OneMinusDstAlpha,
                            dst_alpha_blend_factor: BlendFactor::One,
                            alpha_blend_op: BlendOp::Add,
                        }),
                        ..Default::default()
                    },
                )),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                    .into_iter()
                    .collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap();

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let allocator = renderer.memory_allocator.clone();
        let buffers = SubbufferAllocator::new(
            allocator.clone(),
            SubbufferAllocatorCreateInfo {
                arena_size: 4 * 1024 * 1024,
                buffer_usage: BufferUsage::VERTEX_BUFFER
                    | BufferUsage::INDEX_BUFFER
                    | BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        Self {
            render_pass,
            pipeline,
            sampler,
            framebuffers: Vec::new(),
            framebuffer_generation: None,
            allocator,
            descriptors: renderer.descriptor_set_allocator(),
            buffers,
            textures: HashMap::new(),
            pending_deltas: Vec::new(),
            clear,
            target_is_srgb: format.numeric_format_color()
                == Some(vulkano::format::NumericFormat::SRGB),
            extent: renderer.swapchain_extent(),
            capture_request: None,
            capture_pending: None,
        }
    }

    fn make_set(&self, view: &Arc<ImageView>) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            self.descriptors.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                view.clone(),
                self.sampler.clone(),
            )],
            [],
        )
        .unwrap()
    }

    /// Show a renderer image in the UI under the given texture id.
    pub fn set_user_texture(&mut self, id: u64, view: Arc<ImageView>) {
        let set = self.make_set(&view);
        self.textures.insert(
            TextureId::User(id),
            UiTexture {
                view,
                set,
                is_linear: true,
            },
        );
    }

    /// Texture changes are kept until a frame is really drawn: skipped frames (for
    /// example during a swapchain rebuild) must not lose them.
    pub fn queue_textures(&mut self, delta: TexturesDelta) {
        if !delta.set.is_empty() || !delta.free.is_empty() {
            self.pending_deltas.push(delta);
        }
    }

    pub fn request_screenshot(&mut self, path: String) {
        self.capture_request = Some(path);
    }

    /// Write out a screenshot once its frame has finished. Frames are waited on in
    /// `Renderer::finish_with`, so anything recorded last frame is already complete.
    pub fn finish_screenshot(&mut self) -> Option<Result<String, String>> {
        let (path, buffer, [w, h], format) = self.capture_pending.take()?;
        let data = buffer.read().ok()?;
        let mut pixels = data.to_vec();
        if matches!(format, Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SRGB) {
            for px in pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        for px in pixels.chunks_exact_mut(4) {
            px[3] = 255;
        }
        Some(
            image::save_buffer(&path, &pixels, w, h, image::ColorType::Rgba8)
                .map(|_| path)
                .map_err(|e| e.to_string()),
        )
    }

    /// Rebuild framebuffers if the swapchain changed. Call after `Renderer::start`.
    pub fn prepare(&mut self, renderer: &Renderer) {
        let generation = renderer.swapchain_generation();
        if self.framebuffer_generation == Some(generation) {
            return;
        }
        self.framebuffer_generation = Some(generation);
        self.extent = renderer.swapchain_extent();
        self.framebuffers = renderer
            .swapchain_images()
            .iter()
            .map(|image| {
                Framebuffer::new(
                    self.render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![ImageView::new_default(image.clone()).unwrap()],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect();
    }

    fn apply_delta(
        &mut self,
        commands: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        id: TextureId,
        delta: &ImageDelta,
    ) {
        let ImageData::Color(image) = &delta.image;
        let [w, h] = image.size.map(|v| v as u32);
        let data: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_array()).collect();

        let staging = self.buffers.allocate_slice::<u8>(data.len() as u64).unwrap();
        staging.write().unwrap().copy_from_slice(&data);

        let target = match delta.pos {
            Some(_) => self.textures[&id].view.image().clone(),
            None => {
                let image = Image::new(
                    self.allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::R8G8B8A8_UNORM,
                        extent: [w, h, 1],
                        usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap();
                let view = ImageView::new_default(image.clone()).unwrap();
                let set = self.make_set(&view);
                self.textures.insert(
                    id,
                    UiTexture {
                        view,
                        set,
                        is_linear: false,
                    },
                );
                image
            }
        };

        let [x, y] = delta.pos.unwrap_or([0, 0]).map(|v| v as u32);
        commands
            .copy_buffer_to_image(CopyBufferToImageInfo {
                regions: [BufferImageCopy {
                    image_subresource: target.subresource_layers(),
                    image_offset: [x, y, 0],
                    image_extent: [w, h, 1],
                    ..Default::default()
                }]
                .into_iter()
                .collect(),
                ..CopyBufferToImageInfo::buffer_image(staging, target)
            })
            .unwrap();
    }

    /// Record the UI pass into the frame's command buffer.
    pub fn paint(
        &mut self,
        commands: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        renderer_images: &[Arc<Image>],
        image_index: u32,
        primitives: &[ClippedPrimitive],
        pixels_per_point: f32,
    ) {
        let mut freed = Vec::new();
        for mut delta in std::mem::take(&mut self.pending_deltas) {
            for (id, image_deltas) in &delta.set {
                for image_delta in image_deltas {
                    self.apply_delta(commands, *id, image_delta);
                }
            }
            freed.extend(delta.free.iter().copied());
            delta.clear();
        }

        let framebuffer = self.framebuffers[image_index as usize].clone();
        commands
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![self.clear.then(|| [0.1, 0.11, 0.14, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [self.extent[0] as f32, self.extent[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap();

        let screen_size = [
            self.extent[0] as f32 / pixels_per_point,
            self.extent[1] as f32 / pixels_per_point,
        ];

        for primitive in primitives {
            let Primitive::Mesh(mesh) = &primitive.primitive else {
                continue;
            };
            if mesh.indices.is_empty() {
                continue;
            }
            let Some(texture) = self.textures.get(&mesh.texture_id) else {
                continue;
            };

            let min_x = (primitive.clip_rect.min.x * pixels_per_point).round().max(0.0) as u32;
            let min_y = (primitive.clip_rect.min.y * pixels_per_point).round().max(0.0) as u32;
            let max_x = ((primitive.clip_rect.max.x * pixels_per_point).round().max(0.0) as u32)
                .min(self.extent[0]);
            let max_y = ((primitive.clip_rect.max.y * pixels_per_point).round().max(0.0) as u32)
                .min(self.extent[1]);
            if max_x <= min_x || max_y <= min_y {
                continue;
            }

            let vertices = self
                .buffers
                .allocate_slice::<UiVertex>(mesh.vertices.len() as u64)
                .unwrap();
            {
                let mut mapped = vertices.write().unwrap();
                for (out, v) in mapped.iter_mut().zip(&mesh.vertices) {
                    *out = UiVertex {
                        pos: [v.pos.x, v.pos.y],
                        uv: [v.uv.x, v.uv.y],
                        color: v.color.to_array(),
                    };
                }
            }
            let indices = self
                .buffers
                .allocate_slice::<u32>(mesh.indices.len() as u64)
                .unwrap();
            indices.write().unwrap().copy_from_slice(&mesh.indices);

            let index_count = mesh.indices.len() as u32;
            commands
                .set_scissor(
                    0,
                    [Scissor {
                        offset: [min_x, min_y],
                        extent: [max_x - min_x, max_y - min_y],
                    }]
                    .into_iter()
                    .collect(),
                )
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    texture.set.clone(),
                )
                .unwrap()
                .push_constants(
                    self.pipeline.layout().clone(),
                    0,
                    vs::PushConstants {
                        screen_size,
                        texture_is_linear: texture.is_linear as u32,
                        target_is_srgb: self.target_is_srgb as u32,
                    },
                )
                .unwrap()
                .bind_vertex_buffers(0, vertices)
                .unwrap()
                .bind_index_buffer(indices)
                .unwrap();
            // SAFETY: indices come from egui and refer to the vertices bound above.
            unsafe {
                commands.draw_indexed(index_count, 1, 0, 0, 0).unwrap();
            }
        }

        commands.end_render_pass(SubpassEndInfo::default()).unwrap();

        for id in freed {
            self.textures.remove(&id);
        }

        if let Some(path) = self.capture_request.take() {
            let image = renderer_images[image_index as usize].clone();
            let [w, h, _] = image.extent();
            let buffer = Buffer::new_slice::<u8>(
                self.allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                (w * h * 4) as u64,
            )
            .unwrap();
            let format = image.format();
            commands
                .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, buffer.clone()))
                .unwrap();
            self.capture_pending = Some((path, buffer, [w, h], format));
        }
    }
}
