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

layout(location = 0) out vec4 f_color;

float compute_shadow(vec4 frag_pos_light_space) {
    // Perform perspective divide
    vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;

    // Transform to [0,1] range
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    //fragments outside the shadow map are considered lit
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
            proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
            proj_coords.z > 1.0) {
        return 1.0;
    }

    // sampler2DShadow does the depth comparison automatically:
    // returns 1.0 if proj_coords.z <= stored depth (lit)
    // returns 0.0 if proj_coords.z > stored depth (shadow)
    // With linear filtering, we get free PCF (percentage-closer filtering) from hardware
    float shadow = texture(shadow_map, vec3(proj_coords.xy, proj_coords.z));
    return shadow;
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
