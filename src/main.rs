mod raster;
mod transformation;
mod utils;
mod sampling;
mod thresholding;
mod gui;

use gui::app::run_app;

fn main() {
    // You might want to add logging initialization here, e.g., `env_logger::init();`
    pollster::block_on(run_app());
}
