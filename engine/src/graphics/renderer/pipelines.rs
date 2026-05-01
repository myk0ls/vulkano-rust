use std::sync::Arc;

use vulkano::{
    device::Device,
    descriptor_set::layout::DescriptorBindingFlags,
    pipeline::{
        ComputePipeline, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
            },
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::{Vertex, VertexDefinition, VertexInputState},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::Subpass,
};
use vulkano::pipeline::graphics::rasterization::DepthBiasState;

use super::{
    SHADOW_MAP_SIZE,
    ambient_frag, ambient_vert,
    ao_comp, blur_comp, composite_frag, composite_vert, cull_comp,
    deferred_frag, deferred_vert,
    directional_frag, directional_vert,
    fxaa_comp,
    light_obj_frag, light_obj_vert,
    pointlight_frag, pointlight_vert,
    shadows_frag, shadows_vert,
    skybox_frag, skybox_vert,
};
use crate::assets::gltf_loader::{ColoredVertex, DummyVertex, NormalVertex};

pub(super) fn create_deferred(
    device: Arc<Device>,
    viewport: Viewport,
    deferred_pass: Subpass,
) -> Arc<GraphicsPipeline> {
    let vs = deferred_vert::load(device.clone()).unwrap();
    let fs = deferred_frag::load(device.clone()).unwrap();
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

    let vertex_input_state = NormalVertex::per_vertex().definition(&vs).unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let mut layout_create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
    if let Some(set1) = layout_create_info.set_layouts.get_mut(1) {
        if let Some(binding) = set1.bindings.get_mut(&1) {
            binding.binding_flags |= DescriptorBindingFlags::PARTIALLY_BOUND
                | DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
            binding.descriptor_count = 65536;
        }
    }

    let layout = PipelineLayout::new(
        device.clone(),
        layout_create_info
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
                deferred_pass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(deferred_pass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

fn additive_blend_state(num_attachments: u32) -> ColorBlendState {
    ColorBlendState::with_attachment_states(
        num_attachments,
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
    )
}

pub(super) fn create_directional(
    device: Arc<Device>,
    viewport: Viewport,
    lighting_pass: Subpass,
) -> Arc<GraphicsPipeline> {
    let vs = directional_vert::load(device.clone()).unwrap();
    let fs = directional_frag::load(device.clone()).unwrap();
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

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

    let num_attachments = lighting_pass.num_color_attachments();
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
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                depth: None,
                ..Default::default()
            }),
            color_blend_state: Some(additive_blend_state(num_attachments)),
            subpass: Some(lighting_pass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

pub(super) fn create_pointlight(
    device: Arc<Device>,
    viewport: Viewport,
    lighting_pass: Subpass,
) -> Arc<GraphicsPipeline> {
    let vs = pointlight_vert::load(device.clone()).unwrap();
    let fs = pointlight_frag::load(device.clone()).unwrap();
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

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

    let num_attachments = lighting_pass.num_color_attachments();
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
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                depth: None,
                ..Default::default()
            }),
            color_blend_state: Some(additive_blend_state(num_attachments)),
            subpass: Some(lighting_pass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

pub(super) fn create_ambient(
    device: Arc<Device>,
    viewport: Viewport,
    lighting_pass: Subpass,
) -> Arc<GraphicsPipeline> {
    let vs = ambient_vert::load(device.clone()).unwrap();
    let fs = ambient_frag::load(device.clone()).unwrap();
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

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

    let num_attachments = lighting_pass.num_color_attachments();
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
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                depth: None,
                ..Default::default()
            }),
            color_blend_state: Some(additive_blend_state(num_attachments)),
            subpass: Some(lighting_pass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

pub(super) fn create_light_obj(
    device: Arc<Device>,
    viewport: Viewport,
    lighting_pass: Subpass,
) -> Arc<GraphicsPipeline> {
    let vs = light_obj_vert::load(device.clone()).unwrap();
    let fs = light_obj_frag::load(device.clone()).unwrap();
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

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

    let num_attachments = lighting_pass.num_color_attachments();
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
                num_attachments,
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(lighting_pass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

pub(super) fn create_skybox(
    device: Arc<Device>,
    viewport: Viewport,
    lighting_pass: Subpass,
) -> Arc<GraphicsPipeline> {
    let vs = skybox_vert::load(device.clone()).unwrap();
    let fs = skybox_frag::load(device.clone()).unwrap();
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

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

    let num_attachments = lighting_pass.num_color_attachments();
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
                    write_enable: false,
                    compare_op: CompareOp::LessOrEqual,
                }),
                depth_bounds: None,
                stencil: None,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                num_attachments,
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(lighting_pass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

pub(super) fn create_shadow(
    device: Arc<Device>,
    shadow_pass: Subpass,
) -> Arc<GraphicsPipeline> {
    let vs = shadows_vert::load(device.clone()).unwrap();
    let fs = shadows_frag::load(device.clone()).unwrap();
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

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
                viewports: [Viewport {
                    offset: [0.0, 0.0],
                    extent: [SHADOW_MAP_SIZE as f32, SHADOW_MAP_SIZE as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Front,
                depth_bias: Some(DepthBiasState {
                    constant_factor: 0.0,
                    clamp: 0.0,
                    slope_factor: 0.0,
                }),
                ..Default::default()
            }),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: None,
            subpass: Some(shadow_pass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

pub(super) fn create_ao(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = ao_comp::load(device.clone()).unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs.entry_point("main").unwrap());
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

pub(super) fn create_blur(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = blur_comp::load(device.clone()).unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs.entry_point("main").unwrap());
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

pub(super) fn create_fxaa(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = fxaa_comp::load(device.clone()).unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs.entry_point("main").unwrap());
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

pub(super) fn create_cull(device: Arc<Device>) -> Arc<ComputePipeline> {
    let cs = cull_comp::load(device.clone()).unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs.entry_point("main").unwrap());
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

pub(super) fn create_composite(
    device: Arc<Device>,
    viewport: Viewport,
    composite_pass: Subpass,
) -> Arc<GraphicsPipeline> {
    let vs = composite_vert::load(device.clone()).unwrap();
    let fs = composite_frag::load(device.clone()).unwrap();
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

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
            vertex_input_state: Some(VertexInputState::new()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                1,
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(composite_pass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}
