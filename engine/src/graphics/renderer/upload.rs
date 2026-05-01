use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyBufferToImageInfo,
        ImageBlit, PrimaryCommandBufferAbstract,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet, layout::DescriptorBindingFlags},
    format::Format,
    image::{
        Image, ImageCreateFlags, ImageCreateInfo, ImageUsage,
        sampler::Filter,
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{Pipeline, PipelineBindPoint},
    sync::GpuFuture,
};

use super::Renderer;
use crate::{
    assets::asset_manager::UnifiedGeometry,
    graphics::{
        mesh::Mesh,
        skybox::{HdrSkyboxImages, Skybox, SkyboxImages},
    },
};

impl Renderer {
    pub fn upload_texture_to_gpu(&self, mesh: &mut Mesh) {
        let mut upload_cmd_buf = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

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
            raw_pixels.clone(),
        )
        .unwrap();

        let image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format,
                extent,
                array_layers,
                mip_levels,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        upload_cmd_buf
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                upload_buffer,
                image.clone(),
            ))
            .unwrap();

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

        if let Some(normal_map) = &mesh.material.normal {
            let img = normal_map.texture.as_ref();
            let raw = img.as_raw();
            let pixels: Vec<u8> = raw.chunks(3).flat_map(|c| [c[0], c[1]]).collect();
            mesh.normal_texture =
                Some(self.upload_raw_to_gpu(pixels, img.width(), img.height(), Format::R8G8_UNORM));
        }

        let has_metallic = mesh.material.pbr.metallic_texture.is_some();
        let has_roughness = mesh.material.pbr.roughness_texture.is_some();
        if has_metallic || has_roughness {
            let (width, height) = mesh
                .material
                .pbr
                .metallic_texture
                .as_ref()
                .or(mesh.material.pbr.roughness_texture.as_ref())
                .map(|t| (t.width(), t.height()))
                .unwrap();
            let pixel_count = (width * height) as usize;
            let m_raw = mesh
                .material
                .pbr
                .metallic_texture
                .as_ref()
                .map(|t| t.as_raw().clone())
                .unwrap_or_else(|| vec![0u8; pixel_count]);
            let r_raw = mesh
                .material
                .pbr
                .roughness_texture
                .as_ref()
                .map(|t| t.as_raw().clone())
                .unwrap_or_else(|| vec![255u8; pixel_count]);
            let pixels: Vec<u8> = m_raw
                .iter()
                .zip(r_raw.iter())
                .flat_map(|(&m, &r)| [m, r])
                .collect();
            mesh.mr_texture =
                Some(self.upload_raw_to_gpu(pixels, width, height, Format::R8G8_UNORM));
        }
    }

    pub fn upload_raw_to_gpu(
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
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
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

        let cube = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent: [512, 512, 1],
                array_layers: 6,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                flags: ImageCreateFlags::CUBE_COMPATIBLE,
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

    pub fn upload_hdr_skybox(&self, hdr: HdrSkyboxImages) -> Skybox {
        let face_size = hdr.face_size;

        let all_bytes: Vec<u8> = hdr
            .faces
            .into_iter()
            .flat_map(|face| face.into_iter().flat_map(|f| f.to_le_bytes()))
            .collect();

        let mut cmd = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let staging = Buffer::from_iter(
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
            all_bytes.into_iter(),
        )
        .unwrap();

        let cube = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: vulkano::image::ImageType::Dim2d,
                format: Format::R32G32B32A32_SFLOAT,
                extent: [face_size, face_size, 1],
                array_layers: 6,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                flags: ImageCreateFlags::CUBE_COMPATIBLE,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        cmd.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, cube.clone()))
            .unwrap();

        let cubemap_view = ImageView::new(
            cube.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Cube,
                ..ImageViewCreateInfo::from_image(&cube)
            },
        )
        .unwrap();

        cmd.build()
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
}
