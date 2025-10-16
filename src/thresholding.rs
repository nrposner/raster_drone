use image::{GrayImage, Luma};


/// Applies Bradley's adaptive thresholding algorithm to a grayscale image.
///
/// This algorithm determines the threshold for each pixel based on the average
/// brightness of a window of pixels around it. This is highly effective for
/// images with varying lighting conditions.
///
/// # Arguments
///
/// * `image` - A reference to the input `GrayImage`.
/// * `s` - The size of the window around each pixel (e.g., a value of 8 means an 8x8 window). 
///   The actual window size used in the calculation will be `s x s`. A good starting
///   point is often `image_width / 8`.
/// * `t` - A percentage threshold. A pixel is set to black if its brightness is `t` percent
///   lower than the average brightness of the surrounding window of pixels.
///   A typical value is 15.
///
/// # Returns
///
/// A new `GrayImage` containing the binarized (black and white) result.
pub fn bradley_adaptive_threshold(image: &GrayImage, s: u32, t: u8) -> GrayImage {
    let (width, height) = image.dimensions();
    let mut output_image = GrayImage::new(width, height);

    // 1. Calculate the integral image.
    // The integral image is a data structure that allows for the rapid calculation
    // of the sum of pixel values in any rectangular area of the image.
    // The value at any point (x, y) in the integral image is the sum of all
    // pixels in the rectangle from (0, 0) to (x, y).
    // We use u64 to prevent overflow for large images.
    let mut integral_image = vec![0u64; (width * height) as usize];

    for y in 0..height {
        let mut row_sum = 0u64;
        for x in 0..width {
            let pixel_value = image.get_pixel(x, y)[0] as u64;
            row_sum += pixel_value;

            let index = (y * width + x) as usize;
            if y == 0 {
                integral_image[index] = row_sum;
            } else {
                let index_above = ((y - 1) * width + x) as usize;
                integral_image[index] = integral_image[index_above] + row_sum;
            }
        }
    }

    // 2. Iterate through each pixel to apply the threshold.
    let s2 = s / 2;

    for y in 0..height {
        for x in 0..width {
            // Define the coordinates of the local window, clamping to image bounds.
            let x1 = x.saturating_sub(s2);
            let x2 = (x + s2).min(width - 1);
            let y1 = y.saturating_sub(s2);
            let y2 = (y + s2).min(height - 1);

            let count = (x2 - x1) * (y2 - y1);

            // Calculate the sum of pixel values in the window using the integral image.
            // This is much faster than summing pixels manually for each window.
            // The sum of a rectangle (x1,y1) to (x2,y2) is:
            // I(x2,y2) - I(x2,y1-1) - I(x1-1,y2) + I(x1-1,y1-1)
            let top_right = integral_image[(y2 * width + x2) as usize];
            let top_left = if x1 > 0 { integral_image[(y2 * width + (x1 - 1)) as usize] } else { 0 };
            let bottom_right = if y1 > 0 { integral_image[((y1 - 1) * width + x2) as usize] } else { 0 };
            let bottom_left = if x1 > 0 && y1 > 0 { integral_image[((y1 - 1) * width + (x1 - 1)) as usize] } else { 0 };

            let sum = top_right + bottom_left - top_left - bottom_right;

            // Apply the thresholding condition.
            let original_pixel_value = image.get_pixel(x, y)[0] as u64;
            let threshold_value = sum * (100 - t as u64) / 100;

            if original_pixel_value * count as u64 <= threshold_value {
                output_image.put_pixel(x, y, Luma([0])); // Black
            } else {
                output_image.put_pixel(x, y, Luma([255])); // White
            }
        }
    }

    output_image
}

