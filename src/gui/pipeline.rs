use std::borrow::Cow;

use image::{DynamicImage, GenericImageView};

use crate::{
    sampling::{farthest_point_sampling, grid_sampling}, 
    thresholding::bradley_adaptive_threshold, 
    transformation::{image_to_coordinates, ImgType},
    raster::SamplingType,
    utils::{Coordinate, CoordinateOutput},
};

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct PreprocessingParams {
    pub img_type: ImgType,
    pub resize: Option<(u32, u32)>,
    pub global_threshold: f32,
    pub use_bradley: bool,
    pub bradley_size: u32,
    pub bradley_threshold: u8,
}

impl Default for PreprocessingParams {
    fn default() -> Self {
        Self {
            img_type: ImgType::BlackOnWhite,
            resize: Some((256, 256)),
            global_threshold: 0.01,
            use_bradley: false,
            bradley_size: 50,
            bradley_threshold: 15,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct SamplingParams {
    pub sample_count: u32,
    sampling_type: SamplingType,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            sample_count: 30,
            sampling_type: SamplingType::Farthest,
        }
    }
}



/// Takes pre-processing params, loads/processes an image, returns all valid coordinates.
pub fn run_preprocessing_stage<'a>(
    params: &PreprocessingParams,
    image: &'a Option<image::DynamicImage>,
) -> Option<CoordinateOutput> {
    // println!("Rerunning EXPENSIVE pre-processing stage...");
    
    // If no image is loaded, there are no coordinates to return.
    let Some(source_img) = image else {
        return None
    };

    // using a CoW pointer to avoid cloning unless necessary down the line
    let mut img_cow: Cow<'a, DynamicImage> = Cow::Borrowed(source_img);

    if params.use_bradley {
        img_cow = Cow::Owned(DynamicImage::ImageLuma8(bradley_adaptive_threshold(
            &img_cow.to_luma8(),
            params.bradley_size,
            params.bradley_threshold,
        )));
    }
    
    if let Some((width, height)) = params.resize {
        // .thumbnail() takes a reference, so we pass our Cow's content.
        img_cow = Cow::Owned(img_cow.thumbnail(width, height));
    }

    let (image_width, image_height) = img_cow.dimensions();

    let initial_coords = image_to_coordinates(&img_cow, params.global_threshold, params.img_type);

    Some(
        CoordinateOutput::new(
            initial_coords,
            image_width,
            image_height,
        )
    )
}


/// Takes sampling params and the full coordinate set, returns the final sample.
pub fn run_sampling_stage(
    params: &SamplingParams,
    intermediate_coords: Option<CoordinateOutput>,
) -> Vec<Coordinate> {
    // println!("Rerunning CHEAP sampling stage...");
    // This is where you would apply your grid, farthest-point, etc., sampling
    // algorithm to the `intermediate_coords`.

    let initial_coords = if let Some(coords) = intermediate_coords {
        coords.coords()
    } else {
        vec![]
    };

    // if the initial coordinates set is less than the supplied number of points,
    // don't sample and just return the whole thing
    if initial_coords.len() <= params.sample_count.try_into().unwrap() {
        initial_coords
    } else {
        match params.sampling_type {
            SamplingType::Farthest => {
                farthest_point_sampling(&initial_coords, params.sample_count)
            },
            SamplingType::Grid => {
                grid_sampling(&initial_coords, params.sample_count)
            }
        }
    }
}
