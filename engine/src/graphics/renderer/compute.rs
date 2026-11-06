use vulkano::format::Format;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassEndInfo, SubpassContents,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    pipeline::{Pipeline, PipelineBindPoint},
};

use super::{Renderer, ao_comp, blur_comp, fxaa_comp};

impl Renderer {
    pub(super) fn dispatch_ao(
        &mut self,
        commands: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let dimensions = self.render_extent();

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

    pub(super) fn dispatch_blur(
        &mut self,
        commands: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let dimensions = self.render_extent();
        let layout = self.blur_pipeline.layout().set_layouts().get(0).unwrap();
        let groups = [(dimensions[0] + 15) / 16, (dimensions[1] + 15) / 16, 1];

        commands
            .bind_pipeline_compute(self.blur_pipeline.clone())
            .unwrap();

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

    pub(super) fn dispatch_fxaa(
        &self,
        commands: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let dimensions = self.render_extent();
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


    /// Run the scene's post effects, each reading the previous one's output.
    pub(super) fn dispatch_post_effects(
        &mut self,
        commands: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        self.post_output = None;
        let effects: Vec<_> = self.post_effects.iter().filter(|e| e.enabled).cloned().collect();
        if effects.is_empty() {
            return;
        }
        let [width, height] = self.render_extent();
        let stale = self
            .post_images
            .as_ref()
            .is_none_or(|images| images[0].image().extent()[..2] != [width, height]);
        if stale {
            let make = || {
                ImageView::new_default(
                    Image::new(
                        self.memory_allocator.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: Format::R16G16B16A16_SFLOAT,
                            extent: [width, height, 1],
                            usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                            ..Default::default()
                        },
                        AllocationCreateInfo::default(),
                    )
                    .unwrap(),
                )
                .unwrap()
            };
            self.post_images = Some([make(), make()]);
        }
        let images = self.post_images.clone().unwrap();
        let time = self.start_time.elapsed().as_secs_f32();

        let mut input = self.fxaa_image.clone();
        let mut next = 0;
        for effect in effects {
            let Some(post) = self.post_pipeline(&effect.path) else {
                continue; // Doesn't compile; the error is reported once.
            };
            let mut params = [0.0; super::custom::MAX_POST_PARAMS];
            for (slot, param) in params.iter_mut().zip(&post.params) {
                *slot = effect.params.get(&param.name).copied().unwrap_or(param.default);
            }
            let output = images[next].clone();
            let layout = post.pipeline.layout().set_layouts()[0].clone();
            let set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                layout,
                [
                    WriteDescriptorSet::image_view_sampler(0, input.clone(), self.clamp_sampler.clone()),
                    WriteDescriptorSet::image_view(1, output.clone()),
                ],
                [],
            )
            .unwrap();
            commands
                .bind_pipeline_compute(post.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(PipelineBindPoint::Compute, post.pipeline.layout().clone(), 0, set)
                .unwrap()
                .push_constants(
                    post.pipeline.layout().clone(),
                    0,
                    super::custom::PostPush {
                        resolution: [width as f32, height as f32],
                        time,
                        _pad: 0.0,
                        params,
                    },
                )
                .unwrap();
            unsafe {
                commands.dispatch([width.div_ceil(16), height.div_ceil(16), 1]).unwrap();
            }
            input = output;
            next = 1 - next;
            self.post_output = Some(input.clone());
        }
    }
}
