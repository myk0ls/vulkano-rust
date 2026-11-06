// A post effect: runs on the whole image after anti-aliasing, before tone mapping.
// Add it under Environment > Post Effects. Colors are linear HDR.
//
// Each "@param name default min max" line becomes a slider in the Inspector and a
// value scripts can change with set_post_param("vignette", "strength", 0.8).
// @param strength 0.6 0 1
// @param tint_red 1.0 0 2

vec4 effect(vec2 uv) {
    vec4 color = scene(uv);
    float edge = smoothstep(0.35, 0.85, length(uv - 0.5));
    color.rgb *= 1.0 - edge * strength;
    color.r *= tint_red;
    return color;
}
