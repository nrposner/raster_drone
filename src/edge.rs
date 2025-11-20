//! Experimenting with edge detection for color images

// in preprocessing

use pyo3::{exceptions::PyValueError, prelude::*};
use edge_detection::canny;

#[allow(dead_code)]
#[pyfunction]
pub fn test_edge(
    input_path: String, 
    output_path: &str,
    sigma: f32,
    strong: f32,
    weak: f32,
) -> PyResult<()> {

    let source_img = match image::open(input_path) {
        Ok(img) => img,
        Err(e) => {
            return Err(PyValueError::new_err(format!("Error loading image: {:?}", e)))
        }
    };

    let detection = canny(
        source_img,
        sigma,  // sigma
        strong,  // strong threshold
        weak, // weak threshold
    );

    let path = std::path::Path::new(output_path);
    if let Some(prefix) = path.parent() {
        std::fs::create_dir_all(prefix).unwrap();
    }

    let output_img = detection.as_image();

    match output_img.save(output_path) {
        Ok(_) => Ok(()),
        Err(e) => Err(PyValueError::new_err(format!("Unable to create file in path 'output/img.png': {}", e)))
    }
}



