use std::sync::Arc;

use vulkano::{
    buffer::allocator::SubbufferAllocator,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{Pipeline, PipelineBindPoint},
};

use super::{
    CulledDrawBuffers, RenderStage, Renderer, ambient_frag, composite_frag, directional_frag,
    directional_vert, shadows_vert, skybox_frag,
};
use crate::{
    assets::asset_manager::{self, UnifiedGeometry},
    graphics::skybox::Skybox,
    scene::components::{
        directional_light::DirectionalLight, pointlight::Pointlight, transform::Transform,
    },
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

        if !self.sun_shadows {
            let (vb, ib) = (
                unified.vertex_buffer.as_ref().unwrap().clone(),
                unified.index_buffer.as_ref().unwrap().clone(),
            );
            let layout = self.shadow_pipeline.layout().set_layouts()[1].clone();
            let draw_data_set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                layout,
                [
                    WriteDescriptorSet::buffer(0, culled.draw_data.clone()),
                    WriteDescriptorSet::buffer(1, culled.joint_matrices.clone()),
                ],
                [],
            )
            .unwrap();
            self.local_shadow_passes(culled, vb, ib, draw_data_set);
            self.begin_main_render_pass();
            return;
        }

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
            [
                WriteDescriptorSet::buffer(0, culled.draw_data.clone()),
                WriteDescriptorSet::buffer(1, culled.joint_matrices.clone()),
            ],
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
            .set_viewport(0, [shadow_viewport(super::SHADOW_MAP_SIZE)].into_iter().collect())
            .unwrap()
            .bind_vertex_buffers(0, vb.clone())
            .unwrap()
            .bind_index_buffer(ib.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.shadow_pipeline.layout().clone(),
                0,
                (light_space_set, draw_data_set.clone()),
            )
            .unwrap();

        unsafe {
            self.commands
                .as_mut()
                .unwrap()
                .draw_indexed_indirect(culled.shadow_indirect.clone())
                .unwrap();
        }

        self.commands
            .as_mut()
            .unwrap()
            .end_render_pass(SubpassEndInfo::default())
            .unwrap();

        self.local_shadow_passes(culled, vb, ib, draw_data_set);
        self.begin_main_render_pass();
    }

    /// Render the scene's depth from each shadow-casting point and spot light.
    fn local_shadow_passes(
        &mut self,
        culled: &CulledDrawBuffers,
        vb: Subbuffer<[crate::assets::gltf_loader::NormalVertex]>,
        ib: Subbuffer<[u32]>,
        draw_data_set: Arc<DescriptorSet>,
    ) {
        let jobs: Vec<(u32, nalgebra_glm::Mat4)> = self
            .local_slots
            .iter()
            .flatten()
            .flat_map(|slot| slot.matrices.iter().enumerate().map(|(i, m)| (slot.layer + i as u32, *m)))
            .collect();
        for (layer, matrix) in jobs {
            let buffer = Buffer::from_data(
                self.memory_allocator.clone(),
                BufferCreateInfo { usage: BufferUsage::UNIFORM_BUFFER, ..Default::default() },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                shadows_vert::LightSpaceMatrix { lightSpaceMatrix: matrix.into() },
            )
            .unwrap();
            let layout = self.shadow_pipeline.layout().set_layouts()[0].clone();
            let set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                layout,
                [WriteDescriptorSet::buffer(0, buffer)],
                [],
            )
            .unwrap();
            let framebuffer = self.local_shadow_framebuffers[layer as usize].clone();
            let commands = self.commands.as_mut().unwrap();
            commands
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some(1.0f32.into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer)
                    },
                    SubpassBeginInfo { contents: SubpassContents::Inline, ..Default::default() },
                )
                .unwrap()
                .bind_pipeline_graphics(self.shadow_pipeline.clone())
                .unwrap()
                .set_viewport(0, [shadow_viewport(super::LOCAL_SHADOW_SIZE)].into_iter().collect())
                .unwrap()
                .bind_vertex_buffers(0, vb.clone())
                .unwrap()
                .bind_index_buffer(ib.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.shadow_pipeline.layout().clone(),
                    0,
                    (set, draw_data_set.clone()),
                )
                .unwrap();
            unsafe {
                commands.draw_indexed_indirect(culled.shadow_indirect.clone()).unwrap();
            }
            commands.end_render_pass(SubpassEndInfo::default()).unwrap();
        }
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
            [
                WriteDescriptorSet::buffer(0, culled.draw_data.clone()),
                WriteDescriptorSet::buffer(1, culled.joint_matrices.clone()),
                WriteDescriptorSet::buffer(2, culled.overrides.clone()),
            ],
            [],
        )
        .unwrap();

        // Material shader variants are compiled (or rebuilt) before recording starts.
        let pipelines: Vec<_> = culled
            .ranges
            .iter()
            .map(|range| {
                range
                    .shader
                    .as_deref()
                    .and_then(|path| self.material_pipeline(path))
                    .unwrap_or_else(|| self.deferred_pipeline.clone())
            })
            .collect();
        let time = self.start_time.elapsed().as_secs_f32();
        let viewport = self.viewport.clone();
        let sets = (
            self.vp_set.clone(),
            self.bindless_material_set.as_ref().unwrap().clone(),
            draw_data_set,
        );
        let builder = self.commands.as_mut().unwrap();

        for (range, pipeline) in culled.ranges.iter().zip(pipelines) {
            builder
                .bind_pipeline_graphics(pipeline.clone())
                .unwrap()
                .set_viewport(0, [viewport.clone()].into_iter().collect())
                .unwrap()
                .bind_vertex_buffers(0, vb.clone())
                .unwrap()
                .bind_index_buffer(ib.clone())
                .unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 0, sets.clone())
                .unwrap()
                .push_constants(
                    pipeline.layout().clone(),
                    0,
                    super::deferred_vert::DeferredPush { draw_offset: range.start, time },
                )
                .unwrap();
            let commands = culled
                .indirect
                .clone()
                .slice(range.start as u64..(range.start + range.count) as u64);
            unsafe {
                builder.draw_indexed_indirect(commands).unwrap();
            }
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
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
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
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
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
                    shadowsEnabled: if self.sun_shadows { 1.0 } else { 0.0 },
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

    /// Draw the point or spot light at `index` of this frame's local lights.
    /// Upload this frame's point and spot lights and sort them into clusters. Runs on the
    /// GPU before the render pass starts, since a compute dispatch can't run inside one.
    pub(super) fn cluster_lights(&mut self) {
        self.cluster_frame = None;
        if self.local_lights.is_empty() {
            return;
        }

        let lights: Vec<super::clustered_frag::Light> = self
            .local_lights
            .iter()
            .zip(&self.local_slots)
            .map(|(light, slot)| {
                let [x, y, z] = light.position;
                let (direction, mut params) = match light.spot {
                    Some((d, inner, outer)) => {
                        let outer = outer.max(0.1);
                        let inner = inner.min(outer - 0.05).max(0.0);
                        ([d[0], d[1], d[2], outer.to_radians().cos()], [inner.to_radians().cos(), 1.0, 0.0, 0.0])
                    }
                    None => ([0.0, 1.0, 0.0, -1.0], [1.0, 0.0, 0.0, 0.0]),
                };
                if let Some(slot) = slot {
                    params[2] = if light.spot.is_some() { 1.0 } else { 2.0 };
                    params[3] = slot.layer as f32;
                }
                let [r, g, b] = light.color;
                super::clustered_frag::Light {
                    position_radius: [x, y, z, light.radius],
                    color_intensity: [r, g, b, light.intensity],
                    direction,
                    params,
                }
            })
            .collect();
        let light_count = lights.len() as u32;

        let mut shadow_matrices = [[[0.0f32; 4]; 4]; 16];
        for slot in self.local_slots.iter().flatten() {
            for (k, m) in slot.matrices.iter().enumerate() {
                if let Some(out) = shadow_matrices.get_mut(slot.layer as usize + k) {
                    *out = (*m).into();
                }
            }
        }

        let upload = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        let lights = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            upload.clone(),
            lights,
        )
        .unwrap();
        let shadow = Buffer::from_data(
            self.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::UNIFORM_BUFFER, ..Default::default() },
            upload.clone(),
            super::clustered_frag::Shadow_Data { matrices: shadow_matrices },
        )
        .unwrap();
        let [width, height] = self.viewport.extent;
        let frame = Buffer::from_data(
            self.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::UNIFORM_BUFFER, ..Default::default() },
            upload,
            super::clustered_frag::Frame_Data {
                view: self.vp.view.into(),
                inv_projection: nalgebra_glm::inverse(&self.vp.projection).into(),
                camera_position: [self.vp.camera_pos.x, self.vp.camera_pos.y, self.vp.camera_pos.z, 1.0],
                screen: [width, height, super::CLUSTER_NEAR, super::CLUSTER_FAR],
                counts: [light_count, 0, 0, 0],
            },
        )
        .unwrap();
        // Only the GPU touches the cluster lists, so they can live in fast device memory.
        let clusters = Buffer::new_slice::<u32>(
            self.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, ..Default::default() },
            (super::CLUSTER_COUNT * super::CLUSTER_SLOT) as u64,
        )
        .unwrap();

        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.light_cluster_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, lights.clone()),
                WriteDescriptorSet::buffer(1, clusters.clone()),
                WriteDescriptorSet::buffer(2, frame.clone()),
            ],
            [],
        )
        .unwrap();
        let commands = self.commands.as_mut().unwrap();
        commands
            .bind_pipeline_compute(self.light_cluster_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.light_cluster_pipeline.layout().clone(), 0, set)
            .unwrap();
        unsafe {
            commands.dispatch([super::CLUSTER_COUNT.div_ceil(64), 1, 1]).unwrap();
        }

        self.cluster_frame = Some(super::ClusterFrame { lights, clusters, frame, shadow });
    }

    /// Shade every point and spot light in one full-screen pass, using the cluster lists.
    pub fn local_lights(&mut self) {
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
        let Some(cluster) = &self.cluster_frame else {
            return;
        };

        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.clustered_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(0, self.color_buffer.clone()),
                WriteDescriptorSet::image_view(1, self.normal_buffer.clone()),
                WriteDescriptorSet::image_view(2, self.frag_location_buffer.clone()),
                WriteDescriptorSet::image_view(3, self.specular_buffer.clone()),
                WriteDescriptorSet::buffer(4, cluster.lights.clone()),
                WriteDescriptorSet::buffer(5, cluster.clusters.clone()),
                WriteDescriptorSet::buffer(6, cluster.frame.clone()),
                WriteDescriptorSet::image_view_sampler(7, self.local_shadow_view.clone(), self.shadow_sampler.clone()),
                WriteDescriptorSet::buffer(8, cluster.shadow.clone()),
            ],
            [],
        )
        .unwrap();

        let commands = self.commands.as_mut().unwrap();
        commands
            .bind_pipeline_graphics(self.clustered_pipeline.clone())
            .unwrap()
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Graphics, self.clustered_pipeline.layout().clone(), 0, set)
            .unwrap();
        unsafe {
            commands.draw(self.dummy_verts.len() as u32, 1, 0, 0).unwrap();
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
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
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
        items: &mut Vec<super::DrawItem>,
        joint_matrices: &[[[f32; 4]; 4]],
        overrides: &[asset_manager::GpuOverride],
    ) -> Option<CulledDrawBuffers> {
        if items.is_empty() {
            return None;
        }

        // Group draws by material shader so each group is one indirect draw.
        items.sort_by(|a, b| a.shader.cmp(&b.shader));
        let mut ranges: Vec<super::DrawRange> = Vec::new();
        for (i, item) in items.iter().enumerate() {
            match ranges.last_mut() {
                Some(range) if range.shader == item.shader => range.count += 1,
                _ => ranges.push(super::DrawRange { shader: item.shader.clone(), start: i as u32, count: 1 }),
            }
        }

        let mut indirect_commands: Vec<DrawIndexedIndirectCommand> =
            Vec::with_capacity(items.len());
        let mut draw_data_vec: Vec<asset_manager::DrawData> = Vec::with_capacity(items.len());
        let mut aabb_vec: Vec<asset_manager::GpuAABB> = Vec::with_capacity(items.len());

        for item in items.iter() {
            let (draw_idx, transform, skin_offset) = (&item.draw_index, &item.transform, &item.skin_offset);
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
                skin_offset: *skin_offset,
                override_index: item.override_index,
                _pad: 0,
            });

            aabb_vec.push(unified.aabb_data[*draw_idx]);
        }

        // Always upload at least one identity matrix so the SSBO binding is valid
        // even when there are no skinned objects (NO_SKIN draws won't index into it).
        let jm_vec: Vec<[[f32; 4]; 4]> = if joint_matrices.is_empty() {
            vec![[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]]
        } else {
            joint_matrices.to_vec()
        };

        let joint_matrices_buffer = Buffer::from_iter(
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
            jm_vec.into_iter(),
        )
        .unwrap();

        let shadow_indirect = Buffer::from_iter(
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
            indirect_commands.iter().copied(),
        )
        .unwrap();

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
            num_draws: items.len() as u32,
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
                .dispatch([(items.len() as u32 + 63) / 64, 1, 1])
                .unwrap();
        }

        let overrides = Buffer::from_iter(
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
            overrides.iter().copied(),
        )
        .unwrap();

        Some(CulledDrawBuffers {
            overrides,
            ranges,
            indirect: indirect_buffer,
            shadow_indirect,
            draw_data: draw_data_buffer,
            joint_matrices: joint_matrices_buffer,
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
                    self.post_output.clone().unwrap_or_else(|| self.fxaa_image.clone()),
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
                        self.composite_framebuffers[self.target_index()].clone(),
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
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
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

fn shadow_viewport(size: u32) -> vulkano::pipeline::graphics::viewport::Viewport {
    vulkano::pipeline::graphics::viewport::Viewport {
        offset: [0.0, 0.0],
        extent: [size as f32, size as f32],
        depth_range: 0.0..=1.0,
    }
}
