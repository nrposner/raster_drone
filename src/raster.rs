use pyo3::{exceptions::PyValueError, prelude::*};
use image::{GrayImage, Luma};

use crate::utils::Coordinate;

/// Creates a new black and white image from a list of coordinates.
///
/// # Arguments
/// * `width` - The width of the new image.
/// * `height` - The height of the new image.
/// * `coords` - A slice of `Coordinate` points to draw in white.
///
/// # Returns
/// A `GrayImage` (grayscale image buffer).
pub fn coordinates_to_image(width: u32, height: u32, coords: &[Coordinate]) -> GrayImage {
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

