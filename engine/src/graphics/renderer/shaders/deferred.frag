#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_location;
layout(location = 3) in vec2 in_tex_coords;
layout(location = 4) flat in uint in_material_index;

layout(set = 1, binding = 0) readonly buffer SpecularArray {
    vec2 data[]; // [intensity, shininess]
} specular_array;

layout(set = 1, binding = 1) uniform sampler2D textures[];

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_normal;
layout(location = 2) out vec4 f_location;
layout(location = 3) out vec2 f_specular;

void main() {
    //f_color = vec4(in_color, 1.0);
    f_color = texture(textures[nonuniformEXT(in_material_index)], in_tex_coords);
    f_normal = vec4(in_normal, 1.0);
    f_location = in_location;
    f_specular = specular_array.data[in_material_index];
}
