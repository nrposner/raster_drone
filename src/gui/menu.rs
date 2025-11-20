use std::{fs::File, io::Write, path::Path};

use rfd::FileDialog;

use crate::{
    gui::app::{AppState}, 
    raster::SamplingType, 
    transformation::ImgType, 
    utils::{Coordinate, ExportCoordinate}
};

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

#[derive(Clone, Debug)]
pub struct SavedCoordinates {
    coords: Vec<Coordinate>,
    color: [f32; 3],
    name: String,
}

#[derive(Clone, Debug)]
pub struct SavedExportCoordinates {
    coords: Vec<ExportCoordinate>,
    color: [f32; 3],
    name: String,
}


#[derive(Clone, Debug)]
pub struct SavedExportCSV {
    csv: Vec<u8>,
    name: String,
}

pub fn save_multiple_csvs(exports: Vec<SavedExportCSV>) -> Result<(), std::io::Error> {
    // 1. Open the folder selection dialog
    let Some(dir_path) = FileDialog::new()
        .set_title("Select folder to save formation CSVs")
        .pick_folder() else {
        println!("Export cancelled by user.");
        return Ok(()); // Return Ok if the user cancels
    };

    // 2. Process each export and write the file
    write_exports_to_directory(&exports, &dir_path)?;

    println!("Successfully exported {} CSV files to: {}", exports.len(), dir_path.display());
    Ok(())
}

fn write_exports_to_directory(exports: &Vec<SavedExportCSV>, dir_path: &Path) -> Result<(), std::io::Error> {
    for export in exports {
        // 1. Construct the full file path: {directory}/{name}.csv
        let file_name = format!("{}.csv", export.name);
        let full_path = dir_path.join(file_name);

        // 2. Open the file for writing
        // This will create the file if it doesn't exist, or truncate it if it does.
        let mut file = File::create(full_path)?;

        // 3. Write the byte vector to the file
        file.write_all(&export.csv)?;
    }
    
    Ok(())
}


// TODO expand this with a Save Formation button
// add a name text box popup?

// saving a formation in preparation to make more
// inefficient as currently writte, but I'm not worried here
// the clones are fine
pub fn ui_save_coordinates_button(ui: &mut egui::Ui, app_state: &mut AppState) {

    if ui.button("Save Coordinates").clicked() {
        // name popup?
        let placeholder = "Placeholder";

        // saving the values
        let saved = match &app_state.saved_light_coords {
            Some(vec) => {

                let new_saved = SavedCoordinates {
                    coords: app_state.final_light_coords.clone(), 
                    color: app_state.visual_params.light_color,
                    name: placeholder.to_string(),
                };
                vec.clone().push(new_saved);
                vec
            },
            None => {
                let new_saved = SavedCoordinates {
                    coords: app_state.final_light_coords.clone(), 
                    color: app_state.visual_params.light_color,
                    name: placeholder.to_string(),
                };
                &vec![new_saved]
            }
        };

        app_state.saved_light_coords = Some(saved.to_vec());
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
                    
                    // we're now going to run this once for every set of coordinates we saved
                    
                    // one clone up front, not in a loop
                    let all_coords_opt = app_state.saved_light_coords.clone();
                    if let Some(all_coords) = all_coords_opt {
                        let normalized_cords: Vec<Result<SavedExportCoordinates, String>> = all_coords.into_iter().map(|coordinates| {

                            // --- Fix for fold initialization ---
                            // Initialize with the first coordinate's values
                            let first = coordinates.coords[0];
                            let (min_x, max_x, min_y, max_y) = coordinates.coords.iter().skip(1).fold(
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
                                Err(format!("The {} formation has only identical coordinates", coordinates.name))
                            } else {
                                let scale_factor = 1.0 / max_range as f64;
                                let new_width = x_space as f64 * scale_factor;
                                let new_height = y_space as f64 * scale_factor;
                                let offset_x = (1.0 - new_width) / 2.0;
                                let offset_y = (1.0 - new_height) / 2.0;

                                let normalized_coordinates: Vec<ExportCoordinate> = coordinates.coords
                                    .iter()
                                    .map(|coord| {
                                        let normalized_x = (coord.x() as f64 - min_x as f64) * scale_factor + offset_x;
                                        // Flip y-axis (1.0 - ...)
                                        let normalized_y = 1.0 - ((coord.y() as f64 - min_y as f64) * scale_factor + offset_y);

                                        // Scale to the final desired dimension
                                        let new_x = normalized_x * max_dim_meters;
                                        let new_y = normalized_y * max_dim_meters;

                                        ExportCoordinate::new(new_x, new_y)
                                    }).collect();

                                Ok(SavedExportCoordinates {
                                    coords: normalized_coordinates,
                                    color: coordinates.color,
                                    name: coordinates.name,
                                })
                            }
                        }).collect();

                        // TODO: and then, we create the new set of CSVs and put them all into a file
                    

                        let all_csvs: Vec<SavedExportCSV> = normalized_cords.iter().map(|coords| {
                            // we create a different one of these CSVs per set of coordinates

                            // --- C. Create the CSV data in memory ---
                            let mut wtr = csv::Writer::from_writer(vec![]);
                            // these need to be converted into u8, normalized on 1

                            // Write header 
                            wtr.write_record([ 
                                "Name", "x_m", "y_m", "z_m", "Red", "Green", "Blue" 
                            ]).unwrap(); // Handle error

                            // a bit messy, we're doing unrelated mutation and return
                            // crutcher would kill me
                            let formation_name = match coords {
                                Ok(valid_coords) => {
                                    // getting colors
                                    let [red, green, blue] = valid_coords.color;
                                    let red_u8 = (red * 255f32) as u8;
                                    let green_u8 = (green * 255f32) as u8;
                                    let blue_u8 = (blue * 255f32) as u8;

                                    for (count, coord) in valid_coords.coords.iter().enumerate() {
                                        wtr.write_record(&[
                                            format!("Drone{}", count+1),
                                             String::from("0.0"),
                                             coord.x().to_string(),
                                             coord.y().to_string(),
                                             red_u8.to_string(),
                                             green_u8.to_string(),
                                             blue_u8.to_string(), 
                                        ]).unwrap(); // todo: Handle error
                                    }
                                    Some(valid_coords.name.clone())
                                },
                                Err(_) => None, 
                            };

                            let csv_data = if let Some(valid_name) = formation_name {

                                // Get the CSV data as bytes
                                match wtr.into_inner() {
                                    Ok(data) => Ok(SavedExportCSV {csv: data, name: valid_name}),
                                    Err(e) => {
                                        app_state.export_error_msg = Some(format!("Formation: {} failed to save", valid_name));
                                        Err(e.to_string())
                                    }
                                }
                            } else { Err("Bla".to_string()) };

                            csv_data

                            // filter out all the empty ones
                        }).filter_map(|v| v.ok() ).collect();


                        // at this point, we move in the file dialog to save all of them

                        // TODO: ignoring result at present
                        let _ = save_multiple_csvs(all_csvs);

                    } else {
                         app_state.export_error_msg = Some("Save some coordinates before exporting!".to_string());
                         return;
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


    egui::ComboBox::from_id_source("sampling type")
        .selected_text(format!("{}", app_state.sampling_params.sampling_type))
        .show_ui(ui, |ui| {
            ui.selectable_value(
                &mut app_state.sampling_params.sampling_type,
                SamplingType::Farthest,
                SamplingType::Farthest.to_string(),
            );
            ui.selectable_value(
                &mut app_state.sampling_params.sampling_type,
                SamplingType::Grid,
                SamplingType::Grid.to_string(),
            );
            ui.selectable_value(
                &mut app_state.sampling_params.sampling_type,
                SamplingType::BlueNoise,
                SamplingType::BlueNoise.to_string(),
            );
        });
    
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

    ui.heading("Save");
    ui_save_coordinates_button(ui, app_state);
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



