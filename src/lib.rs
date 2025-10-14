mod raster;
mod transformation;
mod utils;
mod sampling;
mod thresholding;


// the speed of this application at present means that we may well be able to produce a preview
// application that would allow the user to dynamically alter things like number of drones, provide
// their own image, including possibly color images, and we can cache the initial coordinates, and
// when they change the number of drones, we recalculate the pixel samples on a per-frame basis,
// very smooth feel, could look great to a committee
// would want a nicer 'light' display of drone lights with colors
// maybe against a 'night sky' background instead of a black background??
// maybe the ability to change colors as well? at that point might be a bit too deep into the UI
// part of this

use pyo3::{exceptions::PyValueError, prelude::*};
use image::DynamicImage;

use crate::{raster::{coordinates_to_image, SamplingType}, sampling::{farthest_point_sampling, grid_sampling}, thresholding::bradley_adaptive_threshold, transformation::{image_to_coordinates, ImgType}, utils::Coordinate};

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

    // 4. Turn the sampled coordinates back into an image
    let output_img = coordinates_to_image(
        width,
        height,
        &sampled_coords,
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
pub fn process_image_to_coordinates(
    input_path: String, 
    n: u32, 
    sample: SamplingType, 
    img_type: ImgType,
    resize: Option<(u32, u32)>,
    threshold: f32, 
) -> PyResult<Vec<Coordinate>> {
    let source_img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => {
            return Err(PyValueError::new_err(format!("Error loading image: {:?}", e)))
        }
    };

    let img = if let Some((width, height)) = resize {
        source_img.thumbnail(width, height)
    } else { source_img };

    println!("Image loaded successfully with dimensions: {}x{}", img.width(), img.height());

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

    Ok(sampled_coords)
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
    m.add_function(wrap_pyfunction!(test_bradley, m)?)?;
    Ok(())
}
