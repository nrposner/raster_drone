mod raster;

use pyo3::prelude::*;
use raster::process_image;

/// A Python module implemented in Rust.
#[pymodule]
fn raster_drone(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_image, m)?)?;
    Ok(())
}
