use crate::gui::app::AppState;

#[derive(Debug, PartialEq, Clone, Copy)]
enum ResizeOption {
    None,
    Size256,
    Size512,
    Size1024,
}

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





pub fn populate_slider_menu(app_state: &mut AppState, ui: &mut egui::Ui) {
    ui_load_image_button(ui, app_state);
    // ui.separator();
    // ui.heading("Preprocessing");
    // ui.add(egui::Slider::new(
    //     &mut app_state.preprocessing_params.global_threshold, 
    //     0.0..=1.0
    // ).text("Global Threshold"));

    ui.separator();

    ui.checkbox(
        &mut app_state.preprocessing_params.use_bradley,
        "Use Bradley Thresholding"
    );

    if app_state.preprocessing_params.use_bradley {

        ui.heading("Bradley Thresholding");
        ui.add(egui::Slider::new(
            &mut app_state.preprocessing_params.bradley_threshold,
            1..=100
        ).text("Brightness threshold"));

        ui.heading("Bradley Size");
        ui.add(egui::Slider::new(
            &mut app_state.preprocessing_params.bradley_size,
            1..=200
        ).text("Window Size"));
    }

    ui.separator();

    let mut selected_resize = match app_state.preprocessing_params.resize {
        None => ResizeOption::None,
        Some((256, 256)) => ResizeOption::Size256,
        Some((512, 512)) => ResizeOption::Size512,
        Some((1024, 1024)) => ResizeOption::Size1024,
        _ => ResizeOption::Size256, // our default option
    };

    // Helper to get display text for the selected option.
    let selected_text = match selected_resize {
        ResizeOption::None => "None",
        ResizeOption::Size256 => "256x256",
        ResizeOption::Size512 => "512x512",
        ResizeOption::Size1024 => "1024x1024",
    };

    ui.label("Resize Image");
    egui::ComboBox::from_id_source("resize_combo")
        .selected_text(selected_text)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut selected_resize, ResizeOption::None, "None");
            ui.selectable_value(&mut selected_resize, ResizeOption::Size256, "256x256");
            ui.selectable_value(&mut selected_resize, ResizeOption::Size512, "512x512");
            ui.selectable_value(&mut selected_resize, ResizeOption::Size1024, "1024x1024");
        });

    // 4. After the UI has been drawn, convert the enum back to the data model.
    app_state.preprocessing_params.resize = match selected_resize {
        ResizeOption::None => None,
        ResizeOption::Size256 => Some((256, 256)),
        ResizeOption::Size512 => Some((512, 512)),
        ResizeOption::Size1024 => Some((1024, 1024)),
    };
    
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
}

pub fn populate_upload_menu(app_state: &mut AppState, ui: &mut egui::Ui) {
    ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
        ui.add_space(ui.available_height() * 0.4);
        ui.heading("Drone Light Show Previewer");
        ui.label("Please load an image to begin.");
        ui.add_space(10.0);
        ui_load_image_button(ui, app_state);
    });
}



