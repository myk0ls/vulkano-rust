use std::sync::Arc;

use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{
        Image, ImageCreateFlags, ImageCreateInfo, ImageUsage,
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    sync::GpuFuture,
};

use super::{Renderer, brdf_lut_comp, irradiance_comp, prefilter_comp};
use crate::graphics::skybox::Skybox;

impl Renderer {
    pub fn bake_irradiance_map(&self, env: &Skybox) -> Skybox {
        const IRR_SIZE: u32 = 32;

        let cs = irradiance_comp::load(self.device.clone()).unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs.entry_point("main").unwrap());
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.device.clone())
                .unwrap(),
        )
        .unwrap();
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap();

        let irr_image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: Format::R32G32B32A32_SFLOAT,
                extent: [IRR_SIZE, IRR_SIZE, 1],
                array_layers: 6,
                usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                flags: ImageCreateFlags::CUBE_COMPATIBLE,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        let array_view = ImageView::new(
            irr_image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Dim2dArray,
                ..ImageViewCreateInfo::from_image(&irr_image)
            },
        )
        .unwrap();
        let cube_view = ImageView::new(
            irr_image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Cube,
                ..ImageViewCreateInfo::from_image(&irr_image)
            },
        )
        .unwrap();

        let set_layout = pipeline.layout().set_layouts().get(0).unwrap();
        let desc_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    env.cubemap.clone(),
                    self.sampler.clone(),
                ),
                WriteDescriptorSet::image_view(1, array_view),
            ],
            [],
        )
        .unwrap();

        let mut cmd = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cmd.bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                desc_set,
            )
            .unwrap();

        unsafe {
            cmd.dispatch([IRR_SIZE / 8, IRR_SIZE / 8, 6]).unwrap();
        }

        cmd.build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        Skybox { cubemap: cube_view }
    }

    pub fn bake_prefiltered_env(&self, env: &Skybox) -> Skybox {
        const BASE_SIZE: u32 = 128;
        const NUM_MIPS: u32 = 5;

        let cs = prefilter_comp::load(self.device.clone()).unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs.entry_point("main").unwrap());
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.device.clone())
                .unwrap(),
        )
        .unwrap();
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap();

        let image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: Format::R32G32B32A32_SFLOAT,
                extent: [BASE_SIZE, BASE_SIZE, 1],
                mip_levels: NUM_MIPS,
                array_layers: 6,
                usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                flags: ImageCreateFlags::CUBE_COMPATIBLE,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        let mut cmd = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let set_layout = pipeline.layout().set_layouts().get(0).unwrap();

        for mip in 0..NUM_MIPS {
            let mip_size = (BASE_SIZE >> mip).max(1);
            let roughness = mip as f32 / (NUM_MIPS - 1) as f32;

            let mip_view = ImageView::new(
                image.clone(),
                ImageViewCreateInfo {
                    view_type: ImageViewType::Dim2dArray,
                    subresource_range: vulkano::image::ImageSubresourceRange {
                        aspects: vulkano::image::ImageAspects::COLOR,
                        mip_levels: mip..mip + 1,
                        array_layers: 0..6,
                    },
                    ..ImageViewCreateInfo::from_image(&image)
                },
            )
            .unwrap();

            let desc_set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                set_layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(
                        0,
                        env.cubemap.clone(),
                        self.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view(1, mip_view),
                ],
                [],
            )
            .unwrap();

            let groups = ((mip_size + 7) / 8).max(1);
            cmd.bind_pipeline_compute(pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline.layout().clone(),
                    0,
                    desc_set,
                )
                .unwrap()
                .push_constants(
                    pipeline.layout().clone(),
                    0,
                    prefilter_comp::PushConstants { roughness },
                )
                .unwrap();
            unsafe {
                cmd.dispatch([groups, groups, 6]).unwrap();
            }
        }

        cmd.build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let cube_view = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Cube,
                ..ImageViewCreateInfo::from_image(&image)
            },
        )
        .unwrap();

        Skybox { cubemap: cube_view }
    }

    pub fn bake_brdf_lut(&self) -> Arc<ImageView> {
        const LUT_SIZE: u32 = 512;

        let cs = brdf_lut_comp::load(self.device.clone()).unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs.entry_point("main").unwrap());
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.device.clone())
                .unwrap(),
        )
        .unwrap();
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap();

        let lut_image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: Format::R32G32B32A32_SFLOAT,
                extent: [LUT_SIZE, LUT_SIZE, 1],
                usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        let lut_view = ImageView::new_default(lut_image.clone()).unwrap();

        let set_layout = pipeline.layout().set_layouts().get(0).unwrap();
        let desc_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [WriteDescriptorSet::image_view(0, lut_view.clone())],
            [],
        )
        .unwrap();

        let mut cmd = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cmd.bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                desc_set,
            )
            .unwrap();
        unsafe {
            cmd.dispatch([LUT_SIZE / 8, LUT_SIZE / 8, 1]).unwrap();
        }

        cmd.build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        lut_view
    }
}
