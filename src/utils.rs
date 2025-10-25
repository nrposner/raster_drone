use pyo3::{pyclass, pymethods, IntoPyObject};

use crate::transformation::ColorCoordinate;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, IntoPyObject, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Coordinate([u32; 2]);

impl Coordinate {
    pub fn new(x: u32, y: u32) -> Self {
        Self([x, y])
    }
    pub fn x(&self) -> u32 {
        self.0[0]
    }
    pub fn y(&self) -> u32 {
        self.0[1]
    }
    // since we only use it for comparison, it's more performant to use
    // the square of euclidean distances, so that we avoid
    // an expensive square root operation
    pub fn distance_squared(&self, rhs: &Self) -> f64 {
        let dx = self.x().abs_diff(rhs.x()) as f64;
        let dy = self.y().abs_diff(rhs.y()) as f64;
        // This is equivalent to dx.powi(2) + dy.powi(2)
        dx.mul_add(dx, dy * dy)
    }
}

/// Like a Coordinate except we're putting it in f64 instead for the purposes of exporting to
/// Skybrush and normalizing
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExportCoordinate([f64; 2]);

impl ExportCoordinate {
    pub fn new(x: f64, y: f64) -> Self {
        Self([x, y])
    }
    pub fn x(&self) -> f64 {
        self.0[0]
    }
    pub fn y(&self) -> f64 {
        self.0[1]
    }
}

#[derive(Clone)]
#[pyclass(name="CoordinateOutput", module="raster_drone")]
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
    pub fn borrow_coords(self) -> Vec<Coordinate> {
        self.coords
    }
}

#[pymethods]
impl CoordinateOutput {
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn coords(&self) -> Vec<Coordinate> {
        self.coords.clone()
    }
}

#[derive(Clone)]
#[pyclass(name="ColorCoordinateOutput", module="raster_drone")]
pub struct ColorCoordinateOutput {
    coords: Vec<ColorCoordinate>,
    width: u32,
    height: u32,
}

impl ColorCoordinateOutput {
    pub fn new(
        coords: Vec<ColorCoordinate>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            coords,
            width,
            height,
        }
    }
    pub fn borrow_coords(self) -> Vec<ColorCoordinate> {
        self.coords
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn coords(self) -> Vec<ColorCoordinate> {
        self.coords
    }
}

