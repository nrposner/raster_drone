// This struct must match the layout of our `ShaderUniforms` struct in Rust.
struct Uniforms {
    // The width and height of our display area.
    resolution: vec2<f32>,
    // The global color for all lights.
    light_color: vec3<f32>,
    // The user-controlled radius of the light's glow.
    light_radius: f32,
    // The user-controlled intensity/brightness of the light.
    light_intensity: f32,
    // The number of lights to draw for the current frame.
    light_count: u32,
};

// Bind group 0, binding 0: Our uniform data.
@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// Bind group 0, binding 1: An array of light positions.
// The `read` access mode means the shader can only read from this buffer.
@group(0) @binding(1)
var<storage, read> light_positions: array<vec2<f32>>;

// A simple pass-through vertex shader that generates a full-screen triangle.
// We don't need to pass any vertex data; it calculates the vertices on the fly.
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(in_vertex_index / 2u) * 4.0 - 1.0;
    let y = f32(in_vertex_index % 2u) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}


// This is the main function for our fragment shader.
// It receives the pixel's screen coordinate as input.
@fragment
fn fs_main(@builtin(position) pixel_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // The base color of the "night sky" is black.
    var final_color = vec3(0.0);
    
    // A falloff constant derived from the radius.
    // The 4.6 constant is chosen because exp(-4.6) is ~0.01,
    // meaning the glow effectively ends at the radius.
    let radius_sq = uniforms.light_radius * uniforms.light_radius;

    // By clamping `radius_sq` to a small minimum value, we prevent division by zero
    // and the resulting NaN when the radius is 0.
    let falloff_k = 4.6 / max(radius_sq, 0.0001);


    //let falloff_k = 4.6 / (uniforms.light_radius * uniforms.light_radius);

    // Loop through only the active lights for this frame.
    for (var i: u32 = 0u; i < uniforms.light_count; i = i + 1u) {
        let light_pos = light_positions[i];

        // Calculate the square of the distance.
        // This is a common optimization, as it avoids a costly square root operation.
        let dist_sq = dot(pixel_coord.xy - light_pos, pixel_coord.xy - light_pos);

        // Calculate the light's influence using the Gaussian formula.
        let intensity = exp(-falloff_k * dist_sq);

        // Add this light's contribution to the final pixel color.
        final_color += uniforms.light_color * intensity;
    }

    // Apply the global intensity and return the final color for the pixel.
    // The alpha (transparency) is set to 1.0 for fully opaque.
    return vec4<f32>(final_color * uniforms.light_intensity, 1.0);
}
