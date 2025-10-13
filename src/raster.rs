use std::collections::HashMap;
use std::cmp::Ordering;

use pyo3::{exceptions::PyValueError, prelude::*};
use image::{DynamicImage, GenericImageView, GrayImage, Luma};

// It's good practice to add these "derives" to make your struct
// easier to work with (e.g., for printing, copying, and comparing).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate(pub u32, pub u32);

impl Coordinate {
    /// Calculates the squared Euclidean distance to another coordinate.
    /// This is faster than `distance` because it avoids the square root.
    fn distance_squared(&self, rhs: &Self) -> f64 {
        let dx = self.0.abs_diff(rhs.0) as f64;
        let dy = self.1.abs_diff(rhs.1) as f64;
        // This is equivalent to dx.powi(2) + dy.powi(2)
        dx.mul_add(dx, dy * dy)
    }

    /// Calculates the Euclidean distance. Note the fix from the original:
    /// `abs_diff` is used to prevent panics from `u32` subtraction.
    #[allow(dead_code)] // Included for completeness, but we'll use the squared version.
    fn distance(&self, rhs: &Self) -> f64 {
        self.distance_squared(rhs).sqrt()
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
pub fn image_to_coordinates(img: &DynamicImage, percentile: f32) -> Vec<Coordinate> {
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
    // We use `partial_cmp` for f32 and reverse the comparison to get descending order.
    pixel_brightness_data.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

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


/// Selects `n` points from a given set of pixels using the Farthest Point Sampling algorithm.
///
/// This implementation is O(n * m), where 'n' is the number of points to select
/// and 'm' is the total number of input pixels.
///
/// # Arguments
/// * `pixels` - A slice of `Coordinate` points to sample from.
/// * `n` - The number of points to select.
///
/// # Returns
/// A `Vec<Coordinate>` containing the `n` selected points.
pub fn farthest_point_sampling(pixels: &[Coordinate], n: u32) -> Vec<Coordinate> {
    let n = n as usize;
    let m = pixels.len();

    // --- Handle Edge Cases ---
    if n == 0 || m == 0 {
        return Vec::new();
    }
    // If we need to select all or more pixels than are available, just return a copy.
    if n >= m {
        return pixels.to_vec();
    }

    // --- Initialization ---
    let mut selected_pixels = Vec::with_capacity(n);
    // This will store the minimum *squared* distance from each pixel to the selected set.
    let mut min_sq_distances = vec![f64::INFINITY; m];

    // --- Step 1: Select the starting point ---
    // As requested, we'll start with the last pixel in the input slice.
    let first_pixel_index = m - 1;
    let mut last_selected_pixel = pixels[first_pixel_index];

    selected_pixels.push(last_selected_pixel);
    // Mark this pixel as "selected" by setting its distance to 0.
    min_sq_distances[first_pixel_index] = 0.0;

    // --- Step 2: Iteratively select the remaining n-1 points ---
    for _ in 1..n {
        // Update the minimum distances for all points based on the *last* point we added.
        for (i, p) in pixels.iter().enumerate() {
            // We only need to check points that haven't been selected yet.
            if min_sq_distances[i] > 0.0 {
                let sq_dist = p.distance_squared(&last_selected_pixel);
                // If the distance to our newest point is smaller than the previous minimum, update it.
                min_sq_distances[i] = min_sq_distances[i].min(sq_dist);
            }
        }

        // Find the pixel that is now farthest from the entire selected set.
        // We do this by finding the maximum value in our `min_sq_distances` array.
        let (farthest_index, _) = min_sq_distances
            .iter()
            .enumerate()
            .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
            .expect("Distances should have at least one valid value");

        // Add the new farthest pixel to our selection.
        last_selected_pixel = pixels[farthest_index];
        selected_pixels.push(last_selected_pixel);
        min_sq_distances[farthest_index] = 0.0; // And mark it as selected.
    }

    selected_pixels
}

/// Selects points using a grid-based (voxel hashing) approach.
///
/// This implementation is O(m), where 'm' is the total number of input pixels.
///
/// # Arguments
/// * `pixels` - A slice of `Coordinate` points to sample from.
/// * `cell_size` - The side length of each grid cell. A larger size results
///   in a sparser (fewer points) output.
///
/// # Returns
/// A `Vec<Coordinate>` containing the sampled points.
fn grid_sampling(pixels: &[Coordinate], cell_size: u32) -> Vec<Coordinate> {
    // A cell size of 0 would cause a division by zero panic.
    if cell_size == 0 {
        panic!("cell_size cannot be zero.");
    }
    if pixels.is_empty() {
        return Vec::new();
    }

    // The grid is a map from a cell's coordinate `(cx, cy)` to the
    // representative pixel we've chosen for that cell.
    let mut grid: HashMap<(u32, u32), Coordinate> = HashMap::new();

    for &pixel in pixels {
        let cell_x = pixel.0 / cell_size;
        let cell_y = pixel.1 / cell_size;
        let cell_key = (cell_x, cell_y);

        // The `entry` API is perfect for this. `or_insert` will only
        // run if the key `(cell_x, cell_y)` is not already present.
        // This neatly implements our "first one wins" strategy.
        grid.entry(cell_key).or_insert(pixel);
    }

    // The final set of points is simply all the values we stored in the grid.
    grid.into_values().collect()
}


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


#[pyfunction]
pub fn process_image(path: String, n: u32, threshold: f32) -> PyResult<Vec<Vec<f32>>> {
    let source_img = match image::open(path) {
        Ok(img) => img,
        Err(e) => {
            return Err(PyValueError::new_err(format!("Error loading image: {:?}", e)))
        }
    };

    println!("Image loaded successfully with dimensions: {}x{}", source_img.width(), source_img.height());

    // 2. Convert the brightest pixels to coordinates
    // Let's get all pixels with any brightness for this example.
    let initial_coords = image_to_coordinates(&source_img, threshold);
    println!("Extracted {} initial coordinates.", initial_coords.len());

    // 3. Run a sampling algorithm on the coordinates
    let sampled_coords = grid_sampling(&initial_coords, n);
    println!("Sampled down to {} coordinates using a grid.", sampled_coords.len());

    // 4. Turn the sampled coordinates back into an image
    let output_img = coordinates_to_vec_vec(
        source_img.width(),
        source_img.height(),
        &sampled_coords,
    );

    Ok(output_img)
}
