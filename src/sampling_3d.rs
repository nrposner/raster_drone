#![allow(dead_code)]
// use crate::transformation::ColorCoordinate;
// use crate::utils::Coordinate;
// use std::collections::HashMap;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate3D([u32; 3]);

impl Coordinate3D {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self([x, y, z])
    }
    pub fn x(&self) -> u32 {
        self.0[0]
    }
    pub fn y(&self) -> u32 {
        self.0[1]
    }
    pub fn z(&self) -> u32 {
        self.0[2]
    }
    // since we only use it for comparison, it's more performant to use
    // the square of euclidean distances, so that we avoid
    // an expensive square root operation
    pub fn distance_squared(&self, rhs: &Self) -> f64 {
        let dx = self.x().abs_diff(rhs.x()) as f64;
        let dy = self.y().abs_diff(rhs.y()) as f64;
        let dz = self.z().abs_diff(rhs.z()) as f64;
        // find a more performant way to do this using mul_add
        dx.powi(2) + dy.powi(2) + dz.powi(2)
    }
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
pub fn farthest_point_sampling_3d(
    pixels: &[Coordinate3D], 
    n: u32
) -> Vec<Coordinate3D> {
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
