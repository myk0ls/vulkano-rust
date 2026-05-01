use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    buffer::allocator::SubbufferAllocator,
    command_buffer::{
        AutoCommandBufferBuilder, DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassEndInfo, SubpassContents,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{Pipeline, PipelineBindPoint},
};

use super::{
    CulledDrawBuffers, DirectionalLight, Renderer, RenderStage,
    ambient_frag, composite_frag, directional_frag, directional_vert, pointlight_frag, shadows_vert, skybox_frag,
};
use crate::{
    assets::asset_manager::{self, UnifiedGeometry},
    graphics::skybox::Skybox,
    scene::components::{pointlight::Pointlight, transform::Transform},
};

impl Renderer {
    pub fn shadow_pass(
        &mut self,
        light: &DirectionalLight,
        unified: &UnifiedGeometry,
        culled: Option<&CulledDrawBuffers>,
    ) {
        let culled = match culled {
            None => {
                self.begin_main_render_pass();
                return;
            }
            Some(c) => c,
        };

        let light_space_matrix = Renderer::compute_light_space_matrix(light.position);

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
            [WriteDescriptorSet::buffer(0, culled.draw_data.clone())],
            [],
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some(1.0f32.into())],
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
                .draw_indexed_indirect(culled.indirect.clone())
                .unwrap();
        }

        self.commands
            .as_mut()
            .unwrap()
            .end_render_pass(SubpassEndInfo::default())
            .unwrap();

        self.begin_main_render_pass();
    }

    pub fn geometry(&mut self, unified: &UnifiedGeometry, culled: Option<&CulledDrawBuffers>) {
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

        let culled = match culled {
            None => return,
            Some(c) => c,
        };

        let vb = unified.vertex_buffer.as_ref().unwrap().clone();
        let ib = unified.index_buffer.as_ref().unwrap().clone();

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
            [WriteDescriptorSet::buffer(0, culled.draw_data.clone())],
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
            builder
                .draw_indexed_indirect(culled.indirect.clone())
                .unwrap();
        }
    }

    pub fn ambient(
        &mut self,
        irradiance: &Skybox,
        prefiltered: &Skybox,
        brdf_lut: &Arc<vulkano::image::view::ImageView>,
    ) {
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
                WriteDescriptorSet::image_view(1, self.normal_buffer.clone()),
                WriteDescriptorSet::image_view(2, self.frag_location_buffer.clone()),
                WriteDescriptorSet::image_view(3, self.specular_buffer.clone()),
                WriteDescriptorSet::buffer(4, self.ambient_buffer.clone()),
                WriteDescriptorSet::buffer(
                    5,
                    Buffer::from_data(
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
                        ambient_frag::Camera_Data {
                            position: self.vp.camera_pos.into(),
                        },
                    )
                    .unwrap(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    6,
                    irradiance.cubemap.clone(),
                    self.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    7,
                    prefiltered.cubemap.clone(),
                    self.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    8,
                    brdf_lut.clone(),
                    self.clamp_sampler.clone(),
                ),
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
            Self::generate_directional_buffer(&self.directional_allocator, directional_light);

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

    pub fn pointlight(&mut self, light: &Pointlight) {
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

    pub fn light_object(&mut self, _directional_light: &DirectionalLight) {
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
    }

    fn generate_directional_buffer(
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

    pub fn cull_pass(
        &mut self,
        unified: &UnifiedGeometry,
        objects: &[(usize, Transform)],
    ) -> Option<CulledDrawBuffers> {
        if objects.is_empty() {
            return None;
        }

        let mut indirect_commands: Vec<DrawIndexedIndirectCommand> =
            Vec::with_capacity(objects.len());
        let mut draw_data_vec: Vec<asset_manager::DrawData> = Vec::with_capacity(objects.len());
        let mut aabb_vec: Vec<asset_manager::GpuAABB> = Vec::with_capacity(objects.len());

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

            aabb_vec.push(unified.aabb_data[*draw_idx]);
        }

        let indirect_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDIRECT_BUFFER | BufferUsage::STORAGE_BUFFER,
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

        let aabb_buffer = Buffer::from_iter(
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
            aabb_vec.into_iter(),
        )
        .unwrap();

        let layout = self
            .cull_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap()
            .clone();

        let cull_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout,
            [
                WriteDescriptorSet::buffer(0, indirect_buffer.clone()),
                WriteDescriptorSet::buffer(1, draw_data_buffer.clone()),
                WriteDescriptorSet::buffer(2, aabb_buffer),
            ],
            [],
        )
        .unwrap();

        let vp_matrix = self.vp.projection * self.vp.view;
        let planes = Self::extract_frustum_planes(&vp_matrix);

        let push_constants = super::cull_comp::PushConstants {
            planes,
            num_draws: objects.len() as u32,
        };

        let commands = self.commands.as_mut().unwrap();
        commands
            .bind_pipeline_compute(self.cull_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.cull_pipeline.layout().clone(),
                0,
                cull_set,
            )
            .unwrap()
            .push_constants(self.cull_pipeline.layout().clone(), 0, push_constants)
            .unwrap();

        unsafe {
            commands
                .dispatch([(objects.len() as u32 + 63) / 64, 1, 1])
                .unwrap();
        }

        Some(CulledDrawBuffers {
            indirect: indirect_buffer,
            draw_data: draw_data_buffer,
        })
    }

    pub(super) fn composite(
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
