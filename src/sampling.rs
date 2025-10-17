use crate::transformation::ColorCoordinate;
use crate::utils::Coordinate;
use std::collections::HashMap;
use std::cmp::Ordering;

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
pub fn farthest_point_sampling(
    pixels: &[Coordinate], 
    n: u32
) -> Vec<Coordinate> {
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
pub fn grid_sampling(
    pixels: &[Coordinate], 
    cell_size: u32
) -> Vec<Coordinate> {
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
        let cell_x = pixel.x() / cell_size;
        let cell_y = pixel.y() / cell_size;
        let cell_key = (cell_x, cell_y);

        // The `entry` API is perfect for this. `or_insert` will only
        // run if the key `(cell_x, cell_y)` is not already present.
        // This neatly implements our "first one wins" strategy.
        grid.entry(cell_key).or_insert(pixel);
    }

    // The final set of points is simply all the values we stored in the grid.
    grid.into_values().collect()
}

pub fn color_albedo_sampling(pixels: &[ColorCoordinate], n: u32) -> Vec<ColorCoordinate> {

    // take this bundled coordinate, color, and brightness data
    // and use it to select a subsample

    // naive way: perform regular albedo sampling like earlier using farthest point sampling alone

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






