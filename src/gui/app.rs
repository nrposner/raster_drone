use egui_wgpu::wgpu;
use egui_winit::winit::{self, event::{Event, WindowEvent}, event_loop::EventLoop, window::Window};
use std::sync::Arc;
use egui_wgpu::Renderer as EguiRenderer;
use egui_winit::State as EguiState;

use crate::gui::{menu::{populate_slider_menu, populate_upload_menu}, pipeline::{run_preprocessing_stage, run_sampling_stage, PreprocessingParams, SamplingParams}};
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
    pub resolution: [f32; 2],
    viewport_offset: [f32; 2],
    viewport_size: [f32; 2],
    // Pad from offset 8 to 16 for vec3<f32> alignment in WGSL.
    _padding0: [u32; 2],
    pub light_color: [f32; 3],
    // The following fields are all 4-byte aligned and can follow each other.
    pub light_radius: f32,
    pub light_intensity: f32,
    pub light_count: u32,
    // Pad struct to be a multiple of 16 bytes for uniform buffer binding.
    _padding1: [u32; 2],
}

// --- Tiered Pipeline Parameters ---


#[derive(Debug, PartialEq, Clone, Copy)]
pub struct VisualParams {
    pub light_radius: f32,
    pub light_intensity: f32,
    pub light_color: [f32; 3],
}

impl Default for VisualParams {
    fn default() -> Self {
        Self {
            light_radius: 10.0,
            light_intensity: 1.0,
            light_color: [1.0, 0.8, 0.5], // A warm white/yellow
        }
    }
}

// This struct manages all the wgpu-related state.
struct RenderState<'a> {
    _window: Arc<Window>, // Store the Arc to keep the window alive
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
    async fn new(_window: Arc<Window>) -> Self {
        let size = _window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(_window.clone()).unwrap();
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
            _window,
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
pub struct AppState {
    // --- Parameters ---
    pub preprocessing_params: PreprocessingParams,
    pub cached_preprocessing_params: PreprocessingParams,
    pub sampling_params: SamplingParams,
    pub cached_sampling_params: SamplingParams,
    pub visual_params: VisualParams,

    // --- Data ---
    // The raw image data is now stored in memory after being loaded.
    pub image: Option<image::DynamicImage>,
    pub intermediate_coords: Option<CoordinateOutput>,
    pub final_light_coords: Vec<Coordinate>,
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


pub async fn run_app() {
    // --- Basic Setup ---
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(winit::window::WindowBuilder::new()
        .with_title("Image Light Sampler")
        .with_inner_size(winit::dpi::LogicalSize::new(1024, 720))
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
                        let scale_factor = window.scale_factor() as f32;
                        let raw_input = egui_state.take_egui_input(&window);
                        egui_ctx.begin_frame(raw_input);

                        // This will hold the rect of our main drawing area. We won't use it yet.
                        let mut viewport_rect = egui::Rect::NOTHING;

                        // Create the side panel for controls.
                        egui::SidePanel::right("controls_panel").show(&egui_ctx, |ui| {
                            if app_state.image.is_some() {
                                // Call your refactored slider menu function
                                populate_slider_menu(&mut app_state, ui);
                            } else {
                                // Call your refactored upload menu function
                                populate_upload_menu(&mut app_state, ui);
                            }
                        });

                        // Create the central panel to fill the remaining space.
                        egui::CentralPanel::default()
                            .frame(egui::Frame::none())
                            .show(&egui_ctx, |ui| {
                            // For now, we only get the rectangle. We don't do anything with it.
                            viewport_rect = ui.available_rect_before_wrap();
                            // You can add a temporary println here to see its values:
                            // if app_state.image.is_some() { println!("Viewport: {:?}", viewport_rect); }
                        });

                        // The `light_data` and `uniforms` logic below this should remain
                        // unchanged for now. They will still use the old full-screen logic.


                        // // --- Conditional UI: Show waiting screen or main controls ---
                        // if app_state.image.is_some() {
                        //     create_slider_menu(&mut app_state, &egui_ctx);
                        // } else {
                        //     create_upload_menu(&mut app_state, &egui_ctx);
                        // }
                        
                        let egui_output = egui_ctx.end_frame();
                        egui_state.handle_platform_output(
                            &window, 
                            egui_output.platform_output
                        );
                        
                        // This is the crucial step: handle texture updates *before* tessellating.
                        for (id, image_delta) in &egui_output.textures_delta.set {
                            egui_renderer.update_texture(
                                &render_state.device, 
                                &render_state.queue, 
                                *id, 
                                image_delta
                            );
                        }
                        
                        let paint_jobs = egui_ctx.tessellate(
                            egui_output.shapes, 
                            window.scale_factor() as f32
                        );
                        
                        // Free any textures that egui no longer needs.
                        for id in &egui_output.textures_delta.free {
                            egui_renderer.free_texture(id);
                        }
                        
                        // --- Get Surface Texture for Drawing ---
                        let output_frame = match render_state.surface.get_current_texture() {
                            Ok(frame) => frame,
                            Err(e) => { eprintln!("Dropped frame: {:?}", e); return; }
                        };
                        let output_view = output_frame.texture.create_view(
                            &wgpu::TextureViewDescriptor::default()
                        );

                        // --- Record Rendering Commands ---
                        let mut encoder = render_state.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor::default()
                        );
                        let screen_descriptor = egui_wgpu::ScreenDescriptor {
                            size_in_pixels: [render_state.config.width, render_state.config.height],
                            pixels_per_point: window.scale_factor() as f32,
                        };

                        // --- Conditional Rendering Logic ---
                        if app_state.image.is_some() {
                            // --- Update Pipelines if Params Changed ---
                            let mut force_resample = false;
                            if app_state.preprocessing_params != app_state.cached_preprocessing_params {
                                app_state.intermediate_coords = run_preprocessing_stage(
                                    &app_state.preprocessing_params, 
                                    &app_state.image
                                );
                                app_state.cached_preprocessing_params = app_state.preprocessing_params;
                                force_resample = true;
                            }

                            if force_resample || app_state.sampling_params != app_state.cached_sampling_params {
                                app_state.final_light_coords = run_sampling_stage(
                                    &app_state.sampling_params, 
                                    app_state.intermediate_coords.clone()
                                );
                                app_state.cached_sampling_params = app_state.sampling_params;
                            }

                            // --- Update GPU Buffers for Lights Shader ---
                            let uniforms = ShaderUniforms {
                                resolution: [render_state.size.width as f32, render_state.size.height as f32],
                                viewport_offset: [viewport_rect.min.x * scale_factor, viewport_rect.min.y * scale_factor],
                                viewport_size: [viewport_rect.size().x * scale_factor, viewport_rect.size().y * scale_factor],
                                // viewport_offset: [viewport_rect.min.x, viewport_rect.min.y],
                                // viewport_size: [viewport_rect.size().x, viewport_rect.size().y],
                                _padding0: [0, 0],
                                light_color: app_state.visual_params.light_color,
                                light_radius: app_state.visual_params.light_radius,
                                light_intensity: app_state.visual_params.light_intensity,
                                light_count: app_state.final_light_coords.len() as u32,
                                _padding1: [0, 0],
                            };
                            render_state.queue.write_buffer(
                                &render_state.uniform_buffer, 
                                0, 
                                bytemuck::cast_slice(&[uniforms])
                            );

                            let light_data: Vec<[f32; 2]> = if let Some(coords) = &app_state.intermediate_coords {
                                let (img_w, img_h) = (coords.width() as f32, coords.height() as f32);
                                // let (screen_w, screen_h) = (render_state.size.width as f32, render_state.size.height as f32);

                                app_state.final_light_coords.iter()
                                    .map(|coord| {
                                        // Scale and offset coordinates from image space to our new viewport space
                                        let viewport_phys_min_x = viewport_rect.min.x * scale_factor;
                                        let viewport_phys_min_y = viewport_rect.min.y * scale_factor;
                                        let viewport_phys_width = viewport_rect.width() * scale_factor;
                                        let viewport_phys_height = viewport_rect.height() * scale_factor;

                                        let x = (coord.x() as f32 / img_w) * viewport_phys_width + viewport_phys_min_x;
                                        let y = (coord.y() as f32 / img_h) * viewport_phys_height + viewport_phys_min_y;
                                        // let x = (coord.x() as f32 / img_w) * viewport_rect.width() + viewport_rect.min.x;
                                        // let y = (coord.y() as f32 / img_h) * viewport_rect.height() + viewport_rect.min.y;
                                        [x, y]
                                    })
                                    .collect()
                                // app_state.final_light_coords.iter()
                                //     .map(|coord| {
                                //         // Scale coordinates from image space to screen space
                                //         let x = (coord.x() as f32 / img_w) * screen_w;
                                //         let y = (coord.y() as f32 / img_h) * screen_h;
                                //         [x, y]
                                //     })
                                //     .collect()
                            } else {
                                vec![]
                            };

                            render_state.queue.write_buffer(
                                &render_state.lights_storage_buffer, 
                                0,
                                bytemuck::cast_slice(&light_data)
                            );
                            
                            // --- Render Lights + UI ---
                            egui_renderer.update_buffers(
                                &render_state.device, 
                                &render_state.queue, 
                                &mut encoder, 
                                &paint_jobs, 
                                &screen_descriptor
                            );
                            {
                                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("Main Render Pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &output_view,
                                        resolve_target: None,
                                        ops: wgpu::Operations { 
                                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), 
                                            store: wgpu::StoreOp::Store 
                                        },
                                    })],
                                    depth_stencil_attachment: None, 
                                    timestamp_writes: None, 
                                    occlusion_query_set: None,
                                });
                                render_pass.set_pipeline(&render_state.render_pipeline);
                                render_pass.set_bind_group(0, &render_state.bind_group, &[]);
                                render_pass.draw(0..3, 0..1);
                                egui_renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
                            }
                        } else {
                            // --- Pre-warm Shader and Render Egui Waiting Screen ---
                            let uniforms = ShaderUniforms {
                                resolution: [render_state.size.width as f32, render_state.size.height as f32],

                                viewport_offset: [viewport_rect.min.x * scale_factor, viewport_rect.min.y * scale_factor],
                                viewport_size: [viewport_rect.size().x * scale_factor, viewport_rect.size().y * scale_factor],
                                
                                // viewport_offset: [viewport_rect.min.x, viewport_rect.min.y],
                                // viewport_size: [viewport_rect.size().x, viewport_rect.size().y],
                                _padding0: [0,0],
                                light_color: [0.0, 0.0, 0.0],
                                light_radius: 0.0,
                                light_intensity: 0.0,
                                light_count: 0, // CRUCIAL: Tell shader to do nothing.
                                _padding1: [0,0],
                            };
                            render_state.queue.write_buffer(&render_state.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

                            // --- Render Egui Waiting Screen ONLY ---
                            egui_renderer.update_buffers(
                                &render_state.device, 
                                &render_state.queue, 
                                &mut encoder, 
                                &paint_jobs, 
                                &screen_descriptor
                            );
                            {
                                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("Waiting Screen Pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &output_view,
                                        resolve_target: None,
                                        // Clear with a dark gray instead of black
                                        ops: wgpu::Operations { 
                                            load: wgpu::LoadOp::Clear(
                                                wgpu::Color { r: 0.1, g: 0.1, b: 0.12, a: 1.0 }
                                            ), 
                                            store: wgpu::StoreOp::Store },
                                    })],
                                    depth_stencil_attachment: None, 
                                    timestamp_writes: None, 
                                    occlusion_query_set: None,
                                });
                                egui_renderer.render(
                                    &mut render_pass, 
                                    &paint_jobs, 
                                    &screen_descriptor
                                );
                            }
                        }
                        
                        // --- Submit and Present ---
                        render_state.queue.submit(std::iter::once(encoder.finish()));
                        output_frame.present();
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

