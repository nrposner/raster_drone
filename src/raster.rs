use pyo3::{exceptions::PyValueError, prelude::*};
use image::{GrayImage, Luma};

use crate::utils::Coordinate;
use crate::transformation::{image_to_coordinates, ImgType};
use crate::sampling::{farthest_point_sampling, grid_sampling};

/// Creates a new black and white image from a list of coordinates.
///
/// # Arguments
/// * `width` - The width of the new image.
/// * `height` - The height of the new image.
/// * `coords` - A slice of `Coordinate` points to draw in white.
///
/// # Returns
/// A `GrayImage` (grayscale image buffer).
fn coordinates_to_image(width: u32, height: u32, coords: &[Coordinate]) -> GrayImage {
    // Create a new, all-black grayscale image buffer.
    // `GrayImage` is a type alias for `ImageBuffer<Luma<u8>, Vec<u8>>`.
    let mut img = GrayImage::new(width, height);

    // Define the white pixel value. Luma<u8> has one channel from 0 to 255.
    let white_pixel = Luma([255u8]);

    // Iterate through the coordinates and "paint" a white pixel at each location.
    for coord in coords {
        // A bounds check is good practice to prevent panics.
        if coord.0 < width && coord.1 < height {
            img.put_pixel(coord.0, coord.1, white_pixel);
        }
    }

    img
}


#[expect(unused)]
fn coordinates_to_vec_vec(width: u32, height: u32, coords: &[Coordinate]) -> Vec<Vec<f32>> {
    // Create a 2D vector of size height x width, initialized with 0.0 (black).
    let mut image_vec = vec![vec![0.0f32; width as usize]; height as usize];

    // Iterate through the input coordinates and set the corresponding value to 1.0 (white).
    for coord in coords {
        // A bounds check prevents panics if a coordinate is outside the dimensions.
        if coord.0 < width && coord.1 < height {
            image_vec[coord.1 as usize][coord.0 as usize] = 1.0;
        }
    }

    image_vec
}

#[derive(Clone, Copy)]
pub enum SamplingType {
    Grid,
    Farthest,
}

impl FromPyObject<'_> for SamplingType {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(s) = ob.extract::<&str>() {
            match s.to_lowercase().as_str() {
                "grid" => Ok(Self::Grid),
                "farthest" => Ok(Self::Farthest),
                _ => Err(PyValueError::new_err("The valid values for `sampling` include 'grid' and 'farthest'."))
            }
        } else {
            Ok(Self::Farthest)
        }
    }
}

#[pyfunction(signature=(input_path, n, threshold, sample=SamplingType::Farthest, img_type=ImgType::BlackOnWhite, resize=Some((256, 256)), output_path="output/coordinates.png"))]
/// Processes a black and white image into a sample of coordinate pixels
///
/// Arguments:
///     input_path: str 
///         path to source image
///     n: u32
///         number of pixels to select
///     threshold: f64 
///         brightness threshold that gets counted as a 'white' pixel
///     sample: str
///         selecting type of sampling, either 'grid' or 'farthest'. Defaults to 'farthest'
///     img_type: str 
///         selecting type of image, either 'black_on_white' or 'white_on_black'. Defaults to 'black_on_white'
///     resize: (width: u32, height: u32)
///         maximum dimensions by which to resize the image. Will not be resized to exactly those dimensions, but instead to fit within them. Defaults to width = 256, height = 256. Set to None to prevent resizing
///     output_path: str
///         path where the output coordinates image will be saved. Note that, if the intermediate directories do not exist, they will be created. Defaults to 'output/coordinates.png'
///
pub fn process_image(
    input_path: String, 
    n: u32, 
    threshold: f32, 
    sample: SamplingType, 
    img_type: ImgType,
    resize: Option<(u32, u32)>,
    output_path: &str,
) -> PyResult<()> {
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

    println!("Sampled down to {} coordinates using a grid.", sampled_coords.len());

    // 4. Turn the sampled coordinates back into an image
    let output_img = coordinates_to_image(
        img.width(),
        img.height(),
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
