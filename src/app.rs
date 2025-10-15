use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{EventLoop},
    window::Window,
};

use image::{DynamicImage, GenericImageView};

use crate::{sampling::{farthest_point_sampling, grid_sampling}, thresholding::bradley_adaptive_threshold, transformation::image_to_coordinates, ImgType};
use crate::raster::SamplingType;
use crate::utils::{Coordinate, CoordinateOutput};

// Shader code is embedded directly into the binary for simplicity.
const SHADER_CODE: &str = include_str!("lights.wgsl");
// The maximum number of lights we can send to the GPU.
const MAX_LIGHTS: u64 = 65535;

// This struct defines the data we send to the shader every frame.
// It must match the layout of the `Uniforms` struct in the shader.
// `bytemuck` is used to safely cast this struct to a byte slice for the GPU.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderUniforms {
    resolution: [f32; 2],
    light_color: [f32; 3],
    _padding1: u32, // WGPU requires structs to be aligned to 16 bytes.
    light_radius: f32,
    light_intensity: f32,
    light_count: u32,
    _padding2: u32,
}

// --- Tiered Pipeline Parameters ---

#[derive(Debug, PartialEq, Clone, Copy)]
struct PreprocessingParams {
    img_type: ImgType,
    resize: Option<(u32, u32)>,
    global_threshold: f32,
    use_bradley: bool,
    bradley_size: u32,
    bradley_threshold: u8,
}

impl Default for PreprocessingParams {
    fn default() -> Self {
        Self {
            img_type: ImgType::BlackOnWhite,
            resize: Some((256, 256)),
            global_threshold: 0.5,
            use_bradley: true,
            bradley_size: 50,
            bradley_threshold: 15,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct SamplingParams {
    sample_count: u32,
    sampling_type: SamplingType,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            sample_count: 1000,
            sampling_type: SamplingType::Farthest,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct VisualParams {
    light_radius: f32,
    light_intensity: f32,
    light_color: [f32; 3],
}

impl Default for VisualParams {
    fn default() -> Self {
        Self {
            light_radius: 20.0,
            light_intensity: 1.5,
            light_color: [1.0, 0.8, 0.5], // A warm white/yellow
        }
    }
}


// This struct manages all the wgpu-related state.
struct RenderState<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    lights_storage_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl<'a> RenderState<'a> {
    async fn new(window: &'a Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                }
            )
            .await
            .unwrap();
        
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // --- Shader and Pipeline Setup ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_CODE.into()),
        });

        // --- Buffer and Bind Group Setup ---
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<ShaderUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lights_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lights Storage Buffer"),
            size: MAX_LIGHTS * std::mem::size_of::<[f32; 2]>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights_storage_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"), // Simple pass-through vertex shader
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"), // Our main shader logic
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            uniform_buffer,
            lights_storage_buffer,
            bind_group,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}

// This is the main application state, now with tiered parameters and caching.
struct AppState {
    // --- Parameters ---
    preprocessing_params: PreprocessingParams,
    cached_preprocessing_params: PreprocessingParams,
    sampling_params: SamplingParams,
    cached_sampling_params: SamplingParams,
    visual_params: VisualParams,

    // --- Data ---
    // The raw image data is now stored in memory after being loaded.
    image: Option<image::DynamicImage>,
    intermediate_coords: Vec<[f32; 2]>,
    final_light_coords: Vec<[f32; 2]>,
}

impl AppState {
    fn new() -> Self {
        Self {
            preprocessing_params: PreprocessingParams::default(),
            cached_preprocessing_params: PreprocessingParams::default(),
            sampling_params: SamplingParams::default(),
            cached_sampling_params: SamplingParams::default(),
            visual_params: VisualParams::default(),
            image: None,
            intermediate_coords: Vec::new(),
            final_light_coords: Vec::new(),
        }
    }
}



//==================================================================================
//  PLACEHOLDER STAGE 1: Expensive image processing.
//==================================================================================
/// Takes pre-processing params, loads/processes an image, returns all valid coordinates.
fn run_preprocessing_stage(
    params: &PreprocessingParams,
    image: &Option<image::DynamicImage>,
) -> Option<CoordinateOutput> {
    println!("Rerunning EXPENSIVE pre-processing stage...");
    
    // If no image is loaded, there are no coordinates to return.
    let Some(source_img) = image else {
        return None
    };

    let img = if params.use_bradley {
        &DynamicImage::ImageLuma8(
            bradley_adaptive_threshold(
                &source_img.to_luma8(), 
                params.bradley_size, 
                params.bradley_threshold
            )
        )
    } else { source_img };

    let img = if let Some((width, height)) = params.resize {
        &img.thumbnail(width, height)
    } else { img };

    let (image_width, image_height) = img.dimensions();

    let initial_coords = image_to_coordinates(img, params.global_threshold, params.img_type);

    Some(
        CoordinateOutput::new(
            initial_coords,
            image_width,
            image_height,
        )
    )
}



//==================================================================================
//  PLACEHOLDER STAGE 2: Cheap coordinate sampling.
//==================================================================================
/// Takes sampling params and the full coordinate set, returns the final sample.
fn run_sampling_stage(
    params: &SamplingParams,
    intermediate_coords: Option<CoordinateOutput>,
) -> Vec<Coordinate> {
    println!("Rerunning CHEAP sampling stage...");
    // This is where you would apply your grid, farthest-point, etc., sampling
    // algorithm to the `intermediate_coords`.
    // For this example, we'll just take the first N points.

    let initial_coords = if let Some(coords) = intermediate_coords {
        coords.coords()
    } else {
        vec![]
    };

    // if the initial coordinates set is less than the supplied number of points,
    // don't sample and just return the whole thing
    if initial_coords.len() <= params.sample_count.try_into().unwrap() {
        initial_coords
    } else {
        match params.sampling_type {
            SamplingType::Farthest => {
                farthest_point_sampling(&initial_coords, params.sample_count)
            },
            SamplingType::Grid => {
                grid_sampling(&initial_coords, params.sample_count)
            }
        }
    }
}




