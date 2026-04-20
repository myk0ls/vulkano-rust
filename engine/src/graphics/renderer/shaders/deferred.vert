#version 450
#extension GL_ARB_shader_draw_parameters : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec4 tangent; // xyz = tangent dir, w = handedness (+/-1)

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_location;
layout(location = 3) out vec2 tex_coords;
layout(location = 4) flat out uint out_material_index;
layout(location = 5) out vec3 out_tangent;
layout(location = 6) out float out_tangent_w;

layout(set = 0, binding = 0) uniform VP_Data {
    mat4 view;
    mat4 projection;
} vp_uniforms;

struct DrawData {
    mat4 model;
    mat4 normals;
    uint material_index;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

layout(set = 2, binding = 0) readonly buffer DrawDataBuffer {
    DrawData draws[];
} draw_data;

void main() {
    DrawData d = draw_data.draws[gl_DrawIDARB];
    vec4 frag_pos = d.model * vec4(position, 1.0);
    gl_Position = vp_uniforms.projection * vp_uniforms.view * frag_pos;

    mat3 normal_mat = mat3(transpose(inverse(d.model)));
    out_color = color;
    out_normal = normalize(normal_mat * normal);
    out_tangent = normalize(normal_mat * tangent.xyz);
    out_tangent_w = tangent.w;
    out_location = frag_pos;
    tex_coords = uv;
    out_material_index = d.material_index;
}
