A small image processing utility to turn high-contrast images into coordinates for drone shows.

Basic use: 

Here we take a local image file, teacup.jpg, which is a black and white lineart image of a teacup. We will select up to 100 pixels to serve as a low-resolution representation of the image.

```python
import raster_drone as rd

rd.process_image("teacup.jpg", 100, output_path = 'output/teacup.png')
```

The points are selected using farthest-point sampling. Grid-based sampling and more sophisticated sampling types are under development.

```python
import raster_drone as rd

## UNDER DEVELOPMENT: not yet reliable
rd.process_image("teacup.jpg", 100, sampling = 'grid', output_path = 'output/teacup.png')
```

The default assumes images composed of a background of high-brightness, with the image represented by low-brightness pixels (ie black on white). If the image is instead composed of high-brightness pixels on a low-brightness background, set the img_type kwarg to 'white_on_black'

```python
import raster_drone as rd

rd.process_image("white_teacup.jpg", 100, img_type = 'white_on_black', output_path = 'output/teacup.png')
```

The output image is saved to the output path. If the output path is not set, it will default to the 'output/img.png' path. 

Note that if the intermediary directories do not exist, they will be created automatically.

The same process can be used for color images, using the process_color_image function: 

```python
compare_color_images("MonaLisa.jpg", 10000, "output/monalisa.png", resize=(256, 256), background_color='white')
```

Note that because it is not separating foreground from background, the color processing function takes in fewer keyword arguments, and does not perform global thresholding: the visual appeal of output will depend largely on the input, with more colorful, low-detail images being more effectively preserved by the pointillistic effect.
