use crate::{gui::app::AppState, transformation::ImgType, utils::ExportCoordinate};

const FEET_TO_METERS: f64 = 0.3048;

#[derive(Debug, PartialEq, Clone, Copy)]
enum ResizeOption {
    None,
    Size256,
    Size512,
    Size1024,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ExportUnit {
    Meters,
    Feet,
}

// Implement Display to show it nicely in the ComboBox
impl std::fmt::Display for ExportUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportUnit::Meters => write!(f, "Meters"),
            ExportUnit::Feet => write!(f, "Feet"),
        }
    }
}

/// Helper function defining the button that exports the current coordinates as a static CSV
/// compatible with Skybrush Studio
/// Saves to disk a CSV with the following structure:
/// time(ms), x_m (x in meters), y_m, z_m, Red, Green, Blue
pub fn ui_export_coordinates_button(ui: &mut egui::Ui, app_state: &mut AppState) {
    if ui.button("Export Coordinates to CSV").clicked() {
        app_state.show_export_panel = !app_state.show_export_panel;
        app_state.export_error_msg = None;

    }

    // 2. The Export Panel - only shows if toggled
    if app_state.show_export_panel {
        ui.add_space(5.0);

        // `egui::Frame::group` provides a nice visual separation
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.heading("Export Settings");
            ui.add_space(10.0);

            ui.horizontal(|ui| {
                ui.label("Largest Dimension:");
                // Use a TextEdit for the size
                let size_input = ui.add(
                    egui::TextEdit::singleline(&mut app_state.export_size_str)
                        .desired_width(80.0),
                );
                
                // Show parse errors, if any
                if let Some(err) = &app_state.export_error_msg {
                    if size_input.lost_focus() { // Only show error after user is done editing
                        ui.label(egui::RichText::new(err).color(ui.style().visuals.error_fg_color));
                    }
                }
            });

            // ComboBox for unit selection
            egui::ComboBox::from_label("Units")
                .selected_text(format!("{}", app_state.export_unit))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut app_state.export_unit,
                        ExportUnit::Meters,
                        ExportUnit::Meters.to_string(),
                    );
                    ui.selectable_value(
                        &mut app_state.export_unit,
                        ExportUnit::Feet,
                        ExportUnit::Feet.to_string(),
                    );
                });
            
            ui.add_space(10.0);

            // 3. Panel Buttons (Confirm / Cancel)
            ui.horizontal(|ui| {
                // --- The "Confirm" button ---
                if ui.button("Confirm & Save").clicked() {
                    // --- A. Parse and validate input ---
                    let max_dim_input: f64 = match app_state.export_size_str.parse() {
                        Ok(val) => {
                            app_state.export_error_msg = None;
                            val
                        }
                        Err(e) => {
                            app_state.export_error_msg = Some(format!("Invalid number: {}", e));
                            // Don't proceed if parsing failed
                            return; 
                        }
                    };

                    // Convert to meters if necessary
                    let max_dim_meters = match app_state.export_unit {
                        ExportUnit::Meters => max_dim_input,
                        ExportUnit::Feet => max_dim_input * FEET_TO_METERS,
                    };
                    
                    // --- B. Run all transformation logic ---
                    // This logic is now *inside* the confirm button
                    
                    let coordinates = &app_state.final_light_coords;
                    if coordinates.is_empty() {
                         app_state.export_error_msg = Some("No coordinates to export".to_string());
                         return;
                    }

                    // --- Fix for fold initialization ---
                    // Initialize with the first coordinate's values
                    let first = coordinates[0];
                    let (min_x, max_x, min_y, max_y) = coordinates.iter().skip(1).fold(
                        (first.x(), first.x(), first.y(), first.y()),
                        |mut acc, coord| {
                            let x = coord.x();
                            let y = coord.y();
                            if x < acc.0 { acc.0 = x; }
                            if x > acc.1 { acc.1 = x; }
                            if y < acc.2 { acc.2 = y; }
                            if y > acc.3 { acc.3 = y; }
                            acc
                        },
                    );

                    let x_space = max_x - min_x;
                    let y_space = max_y - min_y;
                    let max_range = x_space.max(y_space);

                    // Handle edge case where all points are identical
                    if max_range == 0 {
                        app_state.export_error_msg = Some("All coordinates are identical".to_string());
                        return;
                    }

                    let scale_factor = 1.0 / max_range as f64;
                    let new_width = x_space as f64 * scale_factor;
                    let new_height = y_space as f64 * scale_factor;
                    let offset_x = (1.0 - new_width) / 2.0;
                    let offset_y = (1.0 - new_height) / 2.0;

                    let normalized_coordinates: Vec<ExportCoordinate> = coordinates
                        .iter()
                        .map(|coord| {
                            let normalized_x = (coord.x() as f64 - min_x as f64) * scale_factor + offset_x;
                            // Flip y-axis (1.0 - ...)
                            let normalized_y = 1.0 - ((coord.y() as f64 - min_y as f64) * scale_factor + offset_y);

                            // Scale to the final desired dimension
                            let new_x = normalized_x * max_dim_meters;
                            let new_y = normalized_y * max_dim_meters;

                            ExportCoordinate::new(new_x, new_y)
                        })
                        .collect();
                    
                    // --- C. Create the CSV data in memory ---
                    let mut wtr = csv::Writer::from_writer(vec![]);
                    // these need to be converted into u8, normalized on 1
                    let [red, green, blue] = app_state.visual_params.light_color;
                    let red_u8 = (red * 255f32) as u8;
                    let green_u8 = (green * 255f32) as u8;
                    let blue_u8 = (blue * 255f32) as u8;

                    // Write header
                     wtr.write_record([
                        "Name", "x_m", "y_m", "z_m", "Red", "Green", "Blue"
                     ]).unwrap(); // Handle error

                    for (count, coord) in normalized_coordinates.iter().enumerate() {
                        wtr.write_record(&[
                            format!("Drone{}", count + 1),
                            // String::from("1000"), // setting time in ms to 1 second
                            String::from("0.0"),
                            coord.x().to_string(),
                            coord.y().to_string(),
                            red_u8.to_string(),
                            green_u8.to_string(),
                            blue_u8.to_string(),
                        ])
                        .unwrap(); // todo: Handle error
                    }

                    // Get the CSV data as bytes
                    let csv_data = match wtr.into_inner() {
                        Ok(data) => data,
                        Err(e) => {
                            app_state.export_error_msg = Some(format!("CSV error: {}", e));
                            return;
                        }
                    };

                    // --- D. Open the File Save Dialog ---
                    let file_path = rfd::FileDialog::new()
                        .add_filter("CSV", &["csv"])
                        .set_file_name("skybrush_coords.csv")
                        .save_file();

                    // --- E. Write the file to disk ---
                    if let Some(path) = file_path {
                        match std::fs::write(&path, csv_data) {
                            Ok(_) => {
                                // Success! Hide the panel and clear errors
                                app_state.show_export_panel = false;
                                app_state.export_error_msg = None;
                            }
                            Err(e) => {
                                app_state.export_error_msg = Some(format!("Failed to save file: {}", e));
                            }
                        }
                    } else {
                        // User cancelled the save dialog, just do nothing
                    }
                } // End "Confirm" button

                // --- The "Cancel" button ---
                if ui.button("Cancel").clicked() {
                    app_state.show_export_panel = false;
                    app_state.export_error_msg = None; // Clear any errors
                }
            });
        });
    }
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

    ui.label("Color");
    ui.color_edit_button_rgb(&mut app_state.visual_params.light_color);

    ui.separator();

    let mut selected_contrast = match app_state.preprocessing_params.img_type {
        ImgType::BlackOnWhite => false,
        ImgType::WhiteOnBlack => true,
    };

    ui.checkbox(
        &mut selected_contrast, 
        "Flip Contrast"
    );

    app_state.preprocessing_params.img_type = match selected_contrast {
        false => ImgType::BlackOnWhite,
        true => ImgType::WhiteOnBlack,
    };

    ui.separator();

    ui.heading("Export");
    ui_export_coordinates_button(ui, app_state);
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



