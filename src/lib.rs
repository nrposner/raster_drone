mod raster;
mod transformation;
mod utils;
mod sampling;
mod thresholding;

use pyo3::{exceptions::PyValueError, prelude::*};
use image::DynamicImage;

use crate::{
    raster::{coordinates_to_color_image, coordinates_to_image, BackgroundColor, SamplingType}, 
    sampling::{color_albedo_sampling, farthest_point_sampling, grid_sampling}, 
    thresholding::bradley_adaptive_threshold, 
    transformation::{color_image_to_coordinates, image_to_coordinates, ImgType}, 
    utils::{ColorCoordinateOutput, CoordinateOutput}
};

#[pyfunction(signature=(input_path, n, sample=SamplingType::Farthest, img_type=ImgType::BlackOnWhite, resize=Some((256, 256)), threshold=0.01, bradley=false, bradley_threshold=15, bradley_size=16, output_path="output/coordinates.png"))]
/// Processes a black and white image into a sample of coordinate pixels
///
/// Arguments:
///     input_path: str 
///         path to source image
///     n: u32
///         number of pixels to select
///     sample: str
///         selecting type of sampling, either 'grid' or 'farthest'. Defaults to 'farthest'
///     img_type: str 
///         selecting type of image, either 'black_on_white' or 'white_on_black'. Defaults to 'black_on_white'
///     resize: (width: u32, height: u32)
///         maximum dimensions by which to resize the image. Will not be resized to exactly those dimensions, but instead to fit within them. Defaults to width = 256, height = 256. Set to None to prevent resizing
///     threshold: f64 
///         brightness threshold that gets counted as a 'white' pixel. Defaults to 0.01
///     output_path: str
///         path where the output coordinates image will be saved. Note that, if the intermediate directories do not exist, they will be created. Defaults to 'output/coordinates.png'
#[allow(clippy::too_many_arguments)]
pub fn process_image(
    input_path: String, 
    n: u32, 
    sample: SamplingType, 
    img_type: ImgType,
    resize: Option<(u32, u32)>,
    threshold: f32,
    bradley: bool,
    bradley_threshold: u8,
    bradley_size: u32,
    output_path: &str,
) -> PyResult<()> {

    let coords_output = process_image_to_coordinates(
        input_path, 
        n, 
        sample, 
        img_type, 
        resize, 
        threshold, 
        bradley, 
        bradley_threshold,
        bradley_size
    )?;

    // 4. Turn the sampled coordinates back into an image
    let output_img = coordinates_to_image(
        coords_output.width(),
        coords_output.height(),
        &coords_output.borrow_coords(),
    );

    // creating intermediate directories if necessary
    let path = std::path::Path::new(output_path);
    if let Some(prefix) = path.parent() {
        std::fs::create_dir_all(prefix).unwrap();
    }

    match output_img.save(output_path) {
        Ok(_) => Ok(()),
        Err(e) => Err(PyValueError::new_err(format!("Unable to create file in path 'output/img.png': {}", e)))
    }
}

#[pyfunction(signature=(input_path, n, sample=SamplingType::Farthest, img_type=ImgType::BlackOnWhite, resize=Some((256, 256)), threshold=0.01, bradley=false, bradley_threshold=15, bradley_size=16))]
/// Processes an input image into a vector of (x, y) coordinates
///
/// Arguments:
///     input_path: str 
///         path to source image
///     n: u32
///         number of pixels to select
///     sample: str
///         selecting type of sampling, either 'grid' or 'farthest'. Defaults to 'farthest'
///     img_type: str 
///         selecting type of image, either 'black_on_white' or 'white_on_black'. Defaults to 'black_on_white'
///     resize: (width: u32, height: u32)
///         maximum dimensions by which to resize the image. Will not be resized to exactly those dimensions, but instead to fit within them. Defaults to width = 256, height = 256. Set to None to prevent resizing
///     threshold: f64 
///         brightness threshold that gets counted as a 'white' pixel. Defaults to 0.01
///
/// Returns:
///     coordinates: [(int, int)]
///         the coordinates of each sampled pixel
#[allow(clippy::too_many_arguments)]
pub fn process_image_to_coordinates(
    input_path: String, 
    n: u32, 
    sample: SamplingType, 
    img_type: ImgType,
    resize: Option<(u32, u32)>,
    threshold: f32, 
    bradley: bool,
    bradley_threshold: u8,
    bradley_size: u32,

) -> PyResult<CoordinateOutput> {

    let source_img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => {
            return Err(PyValueError::new_err(format!("Error loading image: {:?}", e)))
        }
    };

    // adding a bradley thresholding step 
    // do we want to apply this before or after resizing the image?
    // let's say after
    let img = if bradley {
        DynamicImage::ImageLuma8(bradley_adaptive_threshold(&source_img.to_luma8(), bradley_size, bradley_threshold))
    } else { source_img };

    let img = if let Some((width, height)) = resize {
        img.thumbnail(width, height)
    } else { img };

    let width = img.width();
    let height = img.height();


    println!("Image loaded successfully with dimensions: {}x{}", width, height);

    // 2. Convert the brightest pixels to coordinates
    // Let's get all pixels with any brightness for this example.
    let initial_coords = image_to_coordinates(&img, threshold, img_type);
    println!("Extracted {} initial coordinates.", initial_coords.len());

    // 3. Run a sampling algorithm on the coordinates
    let sampled_coords = match sample {
        SamplingType::Grid => {
            grid_sampling(&initial_coords, n)
        },
        SamplingType::Farthest => {
            farthest_point_sampling(&initial_coords, n)
        }
    };

    println!("Sampled down to {} coordinates.", sampled_coords.len());

    Ok(
        CoordinateOutput::new(
            sampled_coords,
            width,
            height,
        )
    )
}

#[pyfunction(signature=(input_path, n, resize=Some((256, 256)), background_color="black", output_path="output/coordinates.png"))]
/// Processes a color image into a sample of coordinate pixels
///
/// Arguments:
///     input_path: str 
///         path to source image
///     n: u32
///         number of pixels to select
///     resize: (width: u32, height: u32)
///         maximum dimensions by which to resize the image. Will not be resized to exactly those dimensions, but instead to fit within them. Defaults to width = 256, height = 256. Set to None to prevent resizing
///     background_color: str
///         color of the background pixels not sampled. Options are 'black' or 'white'. Defaults to 'black'
///     output_path: str
///         path where the output coordinates image will be saved. Note that, if the intermediate directories do not exist, they will be created. Defaults to 'output/coordinates.png'
pub fn process_color_image(
    input_path: String, 
    n: u32, 
    resize: Option<(u32, u32)>,
    background_color: &str,
    output_path: &str,
) -> PyResult<()> {

    let coords_output = process_color_image_to_coordinates(
        input_path, 
        n, 
        resize, 
    )?;

    let background_color = match background_color {
        "black" => BackgroundColor::Black,
        "white" => BackgroundColor::White,
        _ => BackgroundColor::Black,
    };

    // 4. Turn the sampled coordinates back into an image
    let output_img = coordinates_to_color_image(
        coords_output.width(),
        coords_output.height(),
        &coords_output.coords(),
        background_color,
    );

    // creating intermediate directories if necessary
    let path = std::path::Path::new(output_path);
    if let Some(prefix) = path.parent() {
        std::fs::create_dir_all(prefix).unwrap();
    }

    match output_img.save(output_path) {
        Ok(_) => Ok(()),
        Err(e) => Err(PyValueError::new_err(format!("Unable to create file in path 'output/img.png': {}", e)))
    }
}

pub fn process_color_image_to_coordinates(
    input_path: String, 
    n: u32, 
    resize: Option<(u32, u32)>,
) -> PyResult<ColorCoordinateOutput> {
    let source_img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => {
            return Err(PyValueError::new_err(format!("Error loading image: {:?}", e)))
        }
    };

    let img = if let Some((width, height)) = resize {
        source_img.thumbnail(width, height)
    } else { source_img };

    let width = img.width();
    let height = img.height();

    let initial_coords = color_image_to_coordinates(&img);

    // sample colors
    let sampled_coords = color_albedo_sampling(&initial_coords, n);

    Ok(ColorCoordinateOutput::new(
        sampled_coords,
        width,
        height,
    ))
}


#[pyfunction(signature=(input_path, size, bradley_threshold=15, output_path="output/bradley.png"))]
fn test_bradley(
    input_path: String, 
    size: u32,
    bradley_threshold: u8,
    output_path: &str,
) -> PyResult<()> {
    let source_img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => {
            return Err(PyValueError::new_err(format!("Error loading image: {:?}", e)))
        }
    };
    let output_img = bradley_adaptive_threshold(&source_img.to_luma8(), size, bradley_threshold);

    // creating intermediate directories if necessary
    let path = std::path::Path::new(output_path);
    if let Some(prefix) = path.parent() {
        std::fs::create_dir_all(prefix).unwrap();
    }

    match output_img.save(output_path) {
        Ok(_) => Ok(()),
        Err(e) => Err(PyValueError::new_err(format!("Unable to create file in path 'output/img.png': {}", e)))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn raster_drone(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_image, m)?)?;
    m.add_function(wrap_pyfunction!(process_color_image, m)?)?;
    m.add_function(wrap_pyfunction!(process_image_to_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(test_bradley, m)?)?;
    Ok(())
}
