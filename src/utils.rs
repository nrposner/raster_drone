use pyo3::IntoPyObject;

#[derive(Debug, Clone, Copy, PartialEq, IntoPyObject)]
pub struct Coordinate(pub u32, pub u32);

impl Coordinate {
    /// Calculates the squared Euclidean distance to another coordinate.
    /// This is faster than `distance` because it avoids the square root.
    pub fn distance_squared(&self, rhs: &Self) -> f64 {
        let dx = self.0.abs_diff(rhs.0) as f64;
        let dy = self.1.abs_diff(rhs.1) as f64;
        // This is equivalent to dx.powi(2) + dy.powi(2)
        dx.mul_add(dx, dy * dy)
    }

    /// Calculates the Euclidean distance. Note the fix from the original:
    /// `abs_diff` is used to prevent panics from `u32` subtraction.
    #[allow(dead_code)] // Included for completeness, but we'll use the squared version.
    pub fn distance(&self, rhs: &Self) -> f64 {
        self.distance_squared(rhs).sqrt()
    }
}

#[derive(Clone, IntoPyObject)]
pub struct CoordinateOutput {
    coords: Vec<Coordinate>,
    width: u32,
    height: u32,
}

impl CoordinateOutput {
    pub fn new(
        coords: Vec<Coordinate>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            coords,
            width,
            height,
        }
    }
    pub fn coords(self) -> Vec<Coordinate> {
        self.coords
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
}
