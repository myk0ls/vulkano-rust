//! Shaders written by the game: material shaders, compiled into variants of the G-buffer
//! shader, and post-processing effects, run as compute shaders after anti-aliasing.
//! Both are compiled at runtime and rebuilt when their file changes.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use vulkano::buffer::BufferContents;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::Subpass;
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};

use super::{Renderer, pipelines};

const DEFERRED_FRAG: &str = include_str!("shaders/deferred.frag");

/// Most post effect parameters a shader may declare.
pub const MAX_POST_PARAMS: usize = 8;

const POST_TEMPLATE_HEAD: &str = "#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) uniform sampler2D u_input;
layout(set = 0, binding = 1, rgba16f) uniform writeonly image2D u_output;
layout(push_constant) uniform PostPush {
    vec2 resolution;
    float time;
    float _pad;
    vec4 params[2];
} post;

// The image so far: linear HDR color, before tone mapping.
vec4 scene(vec2 uv) { return texture(u_input, uv); }
vec4 scene_pixel(ivec2 p) { return texelFetch(u_input, clamp(p, ivec2(0), ivec2(post.resolution) - 1), 0); }
#define RESOLUTION post.resolution
#define TIME post.time
";

const POST_TEMPLATE_TAIL: &str = "
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= int(post.resolution.x) || p.y >= int(post.resolution.y)) return;
    vec2 uv = (vec2(p) + 0.5) / post.resolution;
    imageStore(u_output, p, effect(uv));
}
";

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct PostPush {
    pub resolution: [f32; 2],
    pub time: f32,
    pub _pad: f32,
    pub params: [f32; MAX_POST_PARAMS],
}

/// A tweakable value a post effect declares with `// @param name default min max`.
#[derive(Clone, Debug, PartialEq)]
pub struct PostParam {
    pub name: String,
    pub default: f32,
    pub min: f32,
    pub max: f32,
}

pub fn parse_post_params(source: &str) -> Vec<PostParam> {
    let mut params = Vec::new();
    for line in source.lines() {
        // Only comment lines that start with @param, so prose mentioning it is ignored.
        let Some(rest) = line.trim_start().strip_prefix("//").and_then(|r| r.trim_start().strip_prefix("@param ")) else {
            continue;
        };
        let words: Vec<&str> = rest.split_whitespace().collect();
        let Some(name) = words.first().filter(|n| n.chars().all(|c| c.is_alphanumeric() || c == '_')) else {
            continue;
        };
        let number = |i: usize, fallback: f32| words.get(i).and_then(|w| w.parse().ok()).unwrap_or(fallback);
        let default = number(1, 0.0);
        params.push(PostParam {
            name: name.to_string(),
            default,
            min: number(2, default.min(0.0)),
            max: number(3, default.max(1.0)),
        });
        if params.len() == MAX_POST_PARAMS {
            break;
        }
    }
    params
}

fn compile(source: &str, kind: shaderc::ShaderKind, name: &str) -> Result<Vec<u32>, String> {
    let compiler = shaderc::Compiler::new().ok_or("the shader compiler could not start")?;
    let mut options = shaderc::CompileOptions::new().ok_or("the shader compiler could not start")?;
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);
    let artifact = compiler
        .compile_into_spirv(source, kind, name, "main", Some(&options))
        .map_err(|e| e.to_string().trim().to_string())?;
    Ok(artifact.as_binary().to_vec())
}

/// The G-buffer shader with a material function added.
pub fn material_source(user: &str) -> String {
    DEFERRED_FRAG.replacen("// @material", &format!("#define CUSTOM_MATERIAL\n#line 1\n{user}\n"), 1)
}

/// A post effect's compute shader, with its parameters available by name.
pub fn post_source(user: &str) -> String {
    let mut defines = String::new();
    for (i, param) in parse_post_params(user).iter().enumerate() {
        defines.push_str(&format!("#define {} (post.params[{}][{}])\n", param.name, i / 4, i % 4));
    }
    format!("{POST_TEMPLATE_HEAD}{defines}#line 1\n{user}\n{POST_TEMPLATE_TAIL}")
}

/// Check a material shader file compiles, without a GPU.
pub fn check_material(path: &str) -> Result<(), String> {
    let user = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    compile(&material_source(&user), shaderc::ShaderKind::Fragment, path).map(|_| ())
}

/// Check a post effect file compiles, without a GPU.
pub fn check_post_effect(path: &str) -> Result<(), String> {
    let user = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    compile(&post_source(&user), shaderc::ShaderKind::Compute, path).map(|_| ())
}

fn modified(path: &str) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

struct Cached<T> {
    modified: Option<SystemTime>,
    result: Result<T, String>,
}

pub(super) struct PostPipeline {
    pub pipeline: Arc<ComputePipeline>,
    pub params: Vec<PostParam>,
}

#[derive(Default)]
pub(super) struct CustomShaders {
    materials: HashMap<String, Cached<Arc<GraphicsPipeline>>>,
    posts: HashMap<String, Cached<Arc<PostPipeline>>>,
}

impl CustomShaders {
    pub fn errors(&self) -> Vec<(String, String)> {
        let mut errors: Vec<(String, String)> = self
            .materials
            .iter()
            .filter_map(|(path, c)| c.result.as_ref().err().map(|e| (path.clone(), e.clone())))
            .chain(self.posts.iter().filter_map(|(path, c)| c.result.as_ref().err().map(|e| (path.clone(), e.clone()))))
            .collect();
        errors.sort();
        errors
    }
}

impl Renderer {
    /// The pipeline for a material shader, rebuilt if its file changed. None if it doesn't
    /// compile; the error is in `shader_errors`.
    pub(crate) fn material_pipeline(&mut self, path: &str) -> Option<Arc<GraphicsPipeline>> {
        let modified = modified(path);
        let stale = self.custom.materials.get(path).is_none_or(|c| c.modified != modified);
        if stale {
            let result = self.build_material(path);
            if let Err(e) = &result {
                eprintln!("[shader] {e}");
            }
            self.custom.materials.insert(path.to_string(), Cached { modified, result });
        }
        self.custom.materials[path].result.as_ref().ok().cloned()
    }

    fn build_material(&self, path: &str) -> Result<Arc<GraphicsPipeline>, String> {
        let user = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let words = compile(&material_source(&user), shaderc::ShaderKind::Fragment, path)?;
        // SAFETY: the SPIR-V comes straight from shaderc, which produces valid modules.
        let module = unsafe { ShaderModule::new(self.device.clone(), ShaderModuleCreateInfo::new(&words)) }
            .map_err(|e| format!("{path}: {e}"))?;
        let fs = module.entry_point("main").ok_or("no main function")?;
        let subpass = Subpass::from(self.render_pass.clone(), 0).unwrap();
        // Pipeline creation panics on interface mismatches; report them instead.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pipelines::create_deferred_with(
                self.device.clone(),
                self.viewport.clone(),
                subpass,
                fs,
                Some(self.deferred_pipeline.layout().clone()),
            )
        }))
        .map_err(|_| format!("{path}: the material shader doesn't fit the G-buffer pass"))
    }

    pub(crate) fn post_pipeline(&mut self, path: &str) -> Option<Arc<PostPipeline>> {
        let modified = modified(path);
        let stale = self.custom.posts.get(path).is_none_or(|c| c.modified != modified);
        if stale {
            let result = self.build_post(path);
            if let Err(e) = &result {
                eprintln!("[shader] {e}");
            }
            self.custom.posts.insert(path.to_string(), Cached { modified, result });
        }
        self.custom.posts[path].result.as_ref().ok().cloned()
    }

    fn build_post(&self, path: &str) -> Result<Arc<PostPipeline>, String> {
        let user = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let words = compile(&post_source(&user), shaderc::ShaderKind::Compute, path)?;
        // SAFETY: the SPIR-V comes straight from shaderc, which produces valid modules.
        let module = unsafe { ShaderModule::new(self.device.clone(), ShaderModuleCreateInfo::new(&words)) }
            .map_err(|e| format!("{path}: {e}"))?;
        let stage = PipelineShaderStageCreateInfo::new(module.entry_point("main").ok_or("no main function")?);
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.device.clone())
                .map_err(|e| format!("{path}: {e}"))?,
        )
        .map_err(|e| format!("{path}: {e}"))?;
        let pipeline = ComputePipeline::new(self.device.clone(), None, ComputePipelineCreateInfo::stage_layout(stage, layout))
            .map_err(|e| format!("{path}: {e}"))?;
        Ok(Arc::new(PostPipeline { pipeline, params: parse_post_params(&user) }))
    }

    /// Shaders that failed to compile, with their errors.
    pub fn shader_errors(&self) -> Vec<(String, String)> {
        self.custom.errors()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_params_are_read_from_comments() {
        let params = parse_post_params(
            "// @param strength 0.5 0 2\n// @param radius 0.75\nfloat x; // not a param\n// @param bad-name 1\n// Each \"@param name default\" line is documentation.",
        );
        assert_eq!(
            params,
            [
                PostParam { name: "strength".into(), default: 0.5, min: 0.0, max: 2.0 },
                PostParam { name: "radius".into(), default: 0.75, min: 0.0, max: 1.0 },
            ]
        );
    }

    #[test]
    fn a_valid_material_compiles() {
        let source = material_source(
            "void material(inout Surface s) { s.albedo *= vec3(1.0, 0.5, 0.5); s.emissive = vec3(sin(s.time)); }",
        );
        compile(&source, shaderc::ShaderKind::Fragment, "test.glsl").unwrap();
    }

    #[test]
    fn material_errors_point_at_the_users_line() {
        let source = material_source("void material(inout Surface s) {\n    s.albedo = nope;\n}");
        let error = compile(&source, shaderc::ShaderKind::Fragment, "mine.glsl").unwrap_err();
        assert!(error.contains("mine.glsl:2"), "{error}");
    }

    #[test]
    fn the_base_shader_still_compiles_without_a_material() {
        compile(DEFERRED_FRAG, shaderc::ShaderKind::Fragment, "deferred.frag").unwrap();
    }

    #[test]
    fn a_post_effect_compiles_with_named_params() {
        let source = post_source(
            "// @param strength 0.5 0 1\nvec4 effect(vec2 uv) { vec4 c = scene(uv); return c * (1.0 - strength * length(uv - 0.5)) + TIME * 0.0; }",
        );
        compile(&source, shaderc::ShaderKind::Compute, "vignette.glsl").unwrap();
    }

    #[test]
    fn post_effect_errors_point_at_the_users_line() {
        let source = post_source("vec4 effect(vec2 uv) {\n    return scene(uv) * undefined_thing;\n}");
        let error = compile(&source, shaderc::ShaderKind::Compute, "fx.glsl").unwrap_err();
        assert!(error.contains("fx.glsl:2"), "{error}");
    }
}
