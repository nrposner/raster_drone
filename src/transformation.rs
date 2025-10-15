use pyo3::{exceptions::PyValueError, prelude::*};
use crate::utils::Coordinate;
use image::{DynamicImage, GenericImageView};

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ImgType {
    BlackOnWhite,
    WhiteOnBlack,
}

impl FromPyObject<'_> for ImgType {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(s) = ob.extract::<&str>() {
            match s.to_lowercase().as_str() {
                "black_on_white" => Ok(Self::BlackOnWhite),
                "white_on_black" => Ok(Self::WhiteOnBlack),
                _ => Err(PyValueError::new_err("The valid values for 'img_type' are 'black_on_white' and 'white_on_black'."))
            }
        } else {
            Ok(Self::BlackOnWhite)
        }
    }
}
/// Extracts pixel coordinates from an image based on a brightness percentile.
///
/// # Arguments
/// * `img` - A reference to the image to process.
/// * `percentile` - The fraction of the brightest pixels to select (e.g., 0.1 for the top 10%).
///
/// # Returns
/// A `Vec<Coordinate>` containing the coordinates of the selected pixels.
pub fn image_to_coordinates(img: &DynamicImage, percentile: f32, img_type: ImgType) -> Vec<Coordinate> {
    // Clamp the percentile to a valid range [0.0, 1.0].
    let percentile = percentile.clamp(0.0, 1.0);

    // This buffer will store tuples of (brightness, coordinate) for every pixel.
    let mut pixel_brightness_data = Vec::new();

    // The `pixels()` iterator gives us (x, y, Rgba<u8>) for each pixel.
    for (x, y, pixel) in img.pixels() {
        // `pixel` is Rgba([u8; 4]), where pixel.0 is the array [r, g, b, a].
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let a = pixel[3] as f32;

        // Apply the perceptually weighted luminance formula, multiplied by alpha.
        let brightness = (0.299 * r + 0.587 * g + 0.114 * b) * (a / 255.0);

        // We only care about pixels that have some brightness.
        if brightness > 0.0 {
            pixel_brightness_data.push((brightness, Coordinate(x, y)));
        }
    }

    // --- Sort and Select ---

    // Sort the pixels by brightness in descending order (brightest first).
    // if the desired lines are black on white, we sort normally,
    // if white on black, then we need to reverse the comparison
    match img_type {
        ImgType::BlackOnWhite => pixel_brightness_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()),
        ImgType::WhiteOnBlack => pixel_brightness_data.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()),
    }
    // pixel_brightness_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Calculate how many pixels to take based on the percentile.
    let total_pixels = pixel_brightness_data.len();
    let num_to_take = (total_pixels as f32 * percentile).round() as usize;

    // Take the top `num_to_take` brightest pixels and extract just their coordinates.
    pixel_brightness_data
        .into_iter()
        .take(num_to_take)
        .map(|(_brightness, coord)| coord)
        .collect()
}
