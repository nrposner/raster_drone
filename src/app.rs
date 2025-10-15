use std::borrow::Cow;
use egui_wgpu::wgpu;
use egui_winit::winit::{self, event::{Event, WindowEvent}, event_loop::EventLoop, window::Window};
use std::sync::Arc;
use egui_wgpu::Renderer as EguiRenderer;
use egui_winit::State as EguiState;

use image::{DynamicImage, GenericImageView};

use crate::{sampling::{farthest_point_sampling, grid_sampling}, thresholding::bradley_adaptive_threshold, transformation::{image_to_coordinates, ImgType}};
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
    // 8 bytes of padding is needed here to align `light_color` to a 16-byte boundary.
    // `resolution` is 8 bytes, so we add 8 more to reach 16.
    _padding0: [u32; 2],
    light_color: [f32; 3],
    // `_padding1` correctly aligns the next member to a 4-byte boundary.
    _padding1: u32,
    light_radius: f32,
    light_intensity: f32,
    light_count: u32,
    // The total size of the struct must be a multiple of 16.
    // We are at 44 bytes, so this final padding takes us to 48.
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
    window: Arc<Window>, // Store the Arc to keep the window alive
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
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
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
                },
                None,
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
                entry_point: "vs_main", // Simple pass-through vertex shader
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main", // Our main shader logic
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            window,
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
    intermediate_coords: Option<CoordinateOutput>,
    final_light_coords: Vec<Coordinate>,
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
            intermediate_coords: None,
            final_light_coords: Vec::new(),
        }
    }
}


/// Takes pre-processing params, loads/processes an image, returns all valid coordinates.
fn run_preprocessing_stage<'a>(
    params: &PreprocessingParams,
    image: &'a Option<image::DynamicImage>,
) -> Option<CoordinateOutput> {
    println!("Rerunning EXPENSIVE pre-processing stage...");
    
    // If no image is loaded, there are no coordinates to return.
    let Some(source_img) = image else {
        return None
    };

    // using a CoW pointer to avoid cloning unless necessary down the line
    let mut img_cow: Cow<'a, DynamicImage> = Cow::Borrowed(source_img);

    if params.use_bradley {
        img_cow = Cow::Owned(DynamicImage::ImageLuma8(bradley_adaptive_threshold(
            &img_cow.to_luma8(),
            params.bradley_size,
            params.bradley_threshold,
        )));
    }
    
    if let Some((width, height)) = params.resize {
        // .thumbnail() takes a reference, so we pass our Cow's content.
        img_cow = Cow::Owned(img_cow.thumbnail(width, height));
    }

    let (image_width, image_height) = img_cow.dimensions();

    let initial_coords = image_to_coordinates(&img_cow, params.global_threshold, params.img_type);

    Some(
        CoordinateOutput::new(
            initial_coords,
            image_width,
            image_height,
        )
    )
}


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

/// Helper function to encapsulate the file loading logic.
fn ui_load_image_button(ui: &mut egui::Ui, app_state: &mut AppState) {
    if ui.button("Load Image...").clicked() {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Image Files", &["png", "jpg", "jpeg"])
            .pick_file()
        {
            match image::open(path) {
                Ok(img) => {
                    app_state.image = Some(img);
                    // Invalidate the cache to force the expensive pipeline to re-run on the next frame.
                    // This is a simple way to signal that a major data source has changed.
                    app_state.cached_preprocessing_params.use_bradley = !app_state.preprocessing_params.use_bradley;
                }
                Err(e) => eprintln!("Failed to open image: {}", e),
            }
        }
    }
}

pub async fn run_app() {
    // --- Basic Setup ---
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(winit::window::WindowBuilder::new()
        .with_title("Image Light Sampler")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap());

    // --- State Initialization ---
    let mut render_state = RenderState::new(Arc::clone(&window)).await;
    let mut app_state = AppState::new();

    // --- Egui Setup ---
    let egui_ctx = egui::Context::default();
    let mut egui_state = EguiState::new(
        egui_ctx.clone(),
        egui::ViewportId::ROOT,
        &window,
        None,
        None,
    );
    let mut egui_renderer = EguiRenderer::new(
        &render_state.device,
        render_state.config.format,
        None, // No depth buffer
        1,    // msaa_samples
    );

    // Initial run of the pipelines - will produce empty results since no image is loaded.
    app_state.intermediate_coords = run_preprocessing_stage(&app_state.preprocessing_params, &app_state.image);
    app_state.final_light_coords = run_sampling_stage(&app_state.sampling_params, app_state.intermediate_coords.clone());


    // --- Event Loop ---
    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { window_id, event } if window_id == window.id() => {
                let response = egui_state.on_window_event(&window, &event);
                if response.consumed {
                    return;
                }

                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(physical_size) => {
                        render_state.resize(physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {}
                    WindowEvent::RedrawRequested => {
                        // --- Egui Frame ---
                        let raw_input = egui_state.take_egui_input(&window);
                        egui_ctx.begin_frame(raw_input);

                        // --- Conditional UI: Show waiting screen or main controls ---
                        if app_state.image.is_some() {
                            egui::Window::new("Controls").show(&egui_ctx, |ui| {
                                ui_load_image_button(ui, &mut app_state);
                                ui.separator();
                                ui.heading("Preprocessing");
                                ui.add(egui::Slider::new(&mut app_state.preprocessing_params.global_threshold, 0.0..=1.0).text("Global Threshold"));
                                ui.checkbox(&mut app_state.preprocessing_params.use_bradley, "Use Bradley Thresholding");
                                
                                ui.separator();

                                ui.heading("Sampling");
                                ui.add(egui::Slider::new(&mut app_state.sampling_params.sample_count, 0..=5000).text("Sample Count"));
                                
                                ui.separator();

                                ui.heading("Visuals");
                                ui.add(egui::Slider::new(&mut app_state.visual_params.light_radius, 1.0..=100.0).text("Light Radius"));
                                ui.add(egui::Slider::new(&mut app_state.visual_params.light_intensity, 0.1..=5.0).text("Light Intensity"));
                                ui.color_edit_button_rgb(&mut app_state.visual_params.light_color);
                            });
                        } else {
                            egui::CentralPanel::default().show(&egui_ctx, |ui| {
                                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                    ui.add_space(ui.available_height() * 0.4);
                                    ui.heading("Drone Light Show Previewer");
                                    ui.label("Please load an image to begin.");
                                    ui.add_space(10.0);
                                    ui_load_image_button(ui, &mut app_state);
                                });
                            });
                        }
                        
                        let egui_output = egui_ctx.end_frame();
                        let paint_jobs = egui_ctx.tessellate(egui_output.shapes, window.scale_factor() as f32);
                        
                        // --- Get Surface Texture for Drawing ---
                        let output_frame = match render_state.surface.get_current_texture() {
                            Ok(frame) => frame,
                            Err(e) => { eprintln!("Dropped frame: {:?}", e); return; }
                        };
                        let output_view = output_frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

                        // --- Record Rendering Commands ---
                        let mut encoder = render_state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                        let screen_descriptor = egui_wgpu::ScreenDescriptor {
                            size_in_pixels: [render_state.config.width, render_state.config.height],
                            pixels_per_point: window.scale_factor() as f32,
                        };

                        // --- Conditional Rendering Logic ---
                        if app_state.image.is_some() {
                            // --- Update Pipelines if Params Changed ---
                            let mut force_resample = false;
                            if app_state.preprocessing_params != app_state.cached_preprocessing_params {
                                app_state.intermediate_coords = run_preprocessing_stage(&app_state.preprocessing_params, &app_state.image);
                                app_state.cached_preprocessing_params = app_state.preprocessing_params;
                                force_resample = true;
                            }

                            if force_resample || app_state.sampling_params != app_state.cached_sampling_params {
                                app_state.final_light_coords = run_sampling_stage(&app_state.sampling_params, app_state.intermediate_coords.clone());
                                app_state.cached_sampling_params = app_state.sampling_params;
                            }

                            // --- Update GPU Buffers for Lights Shader ---
                            let uniforms = ShaderUniforms {
                                resolution: [render_state.size.width as f32, render_state.size.height as f32],
                                _padding0: [0, 0],
                                light_color: app_state.visual_params.light_color,
                                _padding1: 0,
                                light_radius: app_state.visual_params.light_radius,
                                light_intensity: app_state.visual_params.light_intensity,
                                light_count: app_state.final_light_coords.len() as u32,
                                _padding2: 0,
                            };
                            render_state.queue.write_buffer(&render_state.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

                            let light_data: Vec<[f32; 2]> = app_state.final_light_coords.iter()
                                .map(|coord| [coord.x() as f32, coord.y() as f32])
                                .collect();
                            render_state.queue.write_buffer(&render_state.lights_storage_buffer, 0, bytemuck::cast_slice(&light_data));
                            
                            // --- Render Lights + UI ---
                            egui_renderer.update_buffers(&render_state.device, &render_state.queue, &mut encoder, &paint_jobs, &screen_descriptor);
                            {
                                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("Main Render Pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &output_view,
                                        resolve_target: None,
                                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                                    })],
                                    depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
                                });
                                render_pass.set_pipeline(&render_state.render_pipeline);
                                render_pass.set_bind_group(0, &render_state.bind_group, &[]);
                                render_pass.draw(0..3, 0..1);
                                egui_renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
                            }
                        } else {
                            // --- Render Egui Waiting Screen ONLY ---
                            egui_renderer.update_buffers(&render_state.device, &render_state.queue, &mut encoder, &paint_jobs, &screen_descriptor);
                            {
                                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("Waiting Screen Pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &output_view,
                                        resolve_target: None,
                                        // Clear with a dark gray instead of black
                                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.12, a: 1.0 }), store: wgpu::StoreOp::Store },
                                    })],
                                    depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
                                });
                                egui_renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
                            }
                        }
                        
                        // --- Submit and Present ---
                        render_state.queue.submit(std::iter::once(encoder.finish()));
                        output_frame.present();
                        egui_state.handle_platform_output(&window, egui_output.platform_output);
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => (),
        }
    })
    .unwrap();
}

