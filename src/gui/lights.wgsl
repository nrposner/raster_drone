// This struct MUST be an exact mirror of the `ShaderUniforms` struct in Rust,
// including the padding fields, to ensure correct memory layout.
struct Uniforms {
    resolution: vec2<f32>,
    viewport_offset: vec2<f32>,
    viewport_size: vec2<f32>,
    // This padding is CRITICAL to align `light_color` to a 16-byte boundary.
    _padding0: vec2<u32>,
    light_color: vec3<f32>,
    light_radius: f32,
    light_intensity: f32,
    light_count: u32,
    // This padding is CRITICAL to make the total struct size a multiple of 16.
    _padding1: vec2<u32>,
};

// Bind group 0, binding 0: Our uniform data.
@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// Bind group 0, binding 1: An array of light positions.
@group(0) @binding(1)
var<storage, read> light_positions: array<vec2<f32>>;

// A simple pass-through vertex shader that generates a full-screen triangle.
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(in_vertex_index / 2u) * 4.0 - 1.0;
    let y = f32(in_vertex_index % 2u) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// The main fragment shader.
@fragment
fn fs_main(@builtin(position) pixel_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // Check if the current pixel is inside our viewport. If not, discard it.
    // This prevents drawing underneath the side panel.
    if (pixel_coord.x < uniforms.viewport_offset.x || 
        pixel_coord.y < uniforms.viewport_offset.y ||
        pixel_coord.x > uniforms.viewport_offset.x + uniforms.viewport_size.x ||
        pixel_coord.y > uniforms.viewport_offset.y + uniforms.viewport_size.y) {
        discard;
    }

    var final_color = vec3(0.0);
    
    let radius_sq = uniforms.light_radius * uniforms.light_radius;
    let falloff_k = 4.6 / max(radius_sq, 0.0001);

    for (var i: u32 = 0u; i < uniforms.light_count; i = i + 1u) {
        let light_pos = light_positions[i];
        
        // Calculate distance between the global pixel coordinate and the global light position.
        let dist_sq = dot(pixel_coord.xy - light_pos, pixel_coord.xy - light_pos);

        let intensity = exp(-falloff_k * dist_sq);
        final_color += uniforms.light_color * intensity;
    }

    return vec4<f32>(final_color * uniforms.light_intensity, 1.0);
}


