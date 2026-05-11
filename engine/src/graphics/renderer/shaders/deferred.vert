#version 450
#extension GL_ARB_shader_draw_parameters : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec4 tangent; // xyz = tangent dir, w = handedness (+/-1)
layout(location = 5) in uvec4 joint_indices;
layout(location = 6) in vec4 joint_weights;

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
    uint skin_offset;
    uint _pad0;
    uint _pad1;
};

layout(set = 2, binding = 0) readonly buffer DrawDataBuffer {
    DrawData draws[];
} draw_data;

layout(set = 2, binding = 1) readonly buffer JointMatrices {
    mat4 matrices[];
} joint_mats;

void main() {
    DrawData d = draw_data.draws[gl_DrawIDARB];

    vec4 local_pos;
    vec3 local_normal;
    vec3 local_tangent;

    if (d.skin_offset != 0xFFFFFFFFu) {
        mat4 skin =
            joint_weights.x * joint_mats.matrices[d.skin_offset + joint_indices.x] +
            joint_weights.y * joint_mats.matrices[d.skin_offset + joint_indices.y] +
            joint_weights.z * joint_mats.matrices[d.skin_offset + joint_indices.z] +
            joint_weights.w * joint_mats.matrices[d.skin_offset + joint_indices.w];
        // Stored positions have Y negated for Vulkan NDC; joint matrices are in glTF Y-up space.
        // Undo the Y-flip before skinning, then re-apply it to stay consistent with static meshes.
        vec4 pos_gltf = vec4(position.x, -position.y, position.z, 1.0);
        vec4 skinned  = skin * pos_gltf;
        local_pos     = vec4(skinned.x, -skinned.y, skinned.z, skinned.w);
        mat3 skin3    = mat3(skin);
        local_normal  = normalize(skin3 * normal);
        local_tangent = normalize(skin3 * tangent.xyz);
    } else {
        local_pos     = vec4(position, 1.0);
        local_normal  = normal;
        local_tangent = tangent.xyz;
    }

    vec4 frag_pos = d.model * local_pos;
    gl_Position = vp_uniforms.projection * vp_uniforms.view * frag_pos;

    mat3 normal_mat = mat3(transpose(inverse(d.model)));
    out_color = color;
    out_normal = normalize(normal_mat * local_normal);
    out_tangent = normalize(normal_mat * local_tangent);
    out_tangent_w = tangent.w;
    out_location = frag_pos;
    tex_coords = uv;
    out_material_index = d.material_index;
}
