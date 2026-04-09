#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_frag_location;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput u_specular;

layout(set = 0, binding = 4) uniform Directional_Light_Data {
    vec4 position;
    vec3 color;
} directional;

layout(set = 0, binding = 5) uniform Camera_Data {
    vec3 position;
} camera;

layout(set = 0, binding = 6) uniform LightSpaceData {
    mat4 light_space_matrix;
} light_space;

layout(set = 0, binding = 7) uniform sampler2DShadow shadow_map;

layout(push_constant) uniform PushConstants {
    float shadowRadius; // PCF kernel radius in shadow-map texels
} pc;

layout(location = 0) out vec4 f_color;

// 16-point Poisson disk, radius ~1.0
const vec2 poissonDisk[16] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554),
    vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507),
    vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367),
    vec2( 0.14383161, -0.14100790)
);

float compute_shadow(vec4 frag_pos_light_space) {
    vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
            proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
            proj_coords.z > 1.0) {
        return 1.0;
    }

    vec2 texel = 1.0 / textureSize(shadow_map, 0);
    float shadow = 0.0;
    for (int i = 0; i < 16; i++) {
        vec2 offset = poissonDisk[i] * texel * pc.shadowRadius;
        shadow += texture(shadow_map, vec3(proj_coords.xy + offset, proj_coords.z));
    }
    return shadow / 16.0;
}

void main() {
    vec3 normal = subpassLoad(u_normals).xyz;
    float specular_intensity = subpassLoad(u_specular).x;
    float specular_shininess = subpassLoad(u_specular).y;
    vec3 view_dir = -normalize(camera.position - subpassLoad(u_frag_location).xyz);
    //vec3 light_direction = normalize(directional.position.xyz + normal);
    vec3 light_direction = normalize(directional.position.xyz);
    vec3 reflect_dir = reflect(-light_direction, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), specular_shininess);
    vec3 specular = specular_intensity * spec * directional.color;
    float directional_intensity = max(dot(normal, light_direction), 0.0);
    vec3 directional_color = directional_intensity * directional.color;

    //shadow
    vec4 frag_pos_light_space = light_space.light_space_matrix * vec4(subpassLoad(u_frag_location).xyz, 1.0);
    float shadow = compute_shadow(frag_pos_light_space);

    vec3 combined_color = shadow * (specular + directional_color) * subpassLoad(u_color).rgb;
    f_color = vec4(combined_color, 1.0);
}
