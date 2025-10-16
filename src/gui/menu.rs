use crate::gui::app::AppState;


/// Helper function to encapsulate the file loading logic.
pub fn ui_load_image_button(ui: &mut egui::Ui, app_state: &mut AppState) {
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


pub fn create_slider_menu(app_state: &mut AppState, egui_ctx: &egui::Context) {
    egui::Window::new("Controls").show(egui_ctx, |ui| {
        ui_load_image_button(ui, app_state);
        ui.separator();
        ui.heading("Preprocessing");
        ui.add(egui::Slider::new(
            &mut app_state.preprocessing_params.global_threshold, 
            0.0..=1.0
        ).text("Global Threshold"));
        ui.checkbox(
            &mut app_state.preprocessing_params.use_bradley,
            "Use Bradley Thresholding"
        );
        
        ui.separator();

        ui.heading("Sampling");
        ui.add(egui::Slider::new(
            &mut app_state.sampling_params.sample_count,
            1..=500
        ).text("Sample Count"));
        
        ui.separator();

        ui.heading("Visuals");
        ui.add(egui::Slider::new(
            &mut app_state.visual_params.light_radius, 
            1.0..=20.0
        ).text("Light Radius"));
        ui.add(egui::Slider::new(
            &mut app_state.visual_params.light_intensity, 
            0.1..=5.0
        ).text("Light Intensity"));
        ui.color_edit_button_rgb(&mut app_state.visual_params.light_color);
    });
}

pub fn create_upload_menu(app_state: &mut AppState, egui_ctx: &egui::Context) {
    egui::CentralPanel::default().show(egui_ctx, |ui| {
        ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
            ui.add_space(ui.available_height() * 0.4);
            ui.heading("Drone Light Show Previewer");
            ui.label("Please load an image to begin.");
            ui.add_space(10.0);
            ui_load_image_button(ui, app_state);
        });
    });
}



