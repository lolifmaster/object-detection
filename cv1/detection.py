from cv1.tools import range
import numpy as np


def in_range(image, lower_bound, upper_bound):
    # Ensure the image is in the correct format (e.g., RGB or BGR)
    if len(image.shape) == 2:
        raise ValueError("Input image must be in color (e.g., RGB or BGR)")

    # Get the height and width of the image
    height, width, _ = image.shape

    # Initialize an output mask with zeros
    mask = np.zeros((height, width), dtype=np.uint8)

    # Extract lower and upper bounds for each channel
    lower_bound_b, lower_bound_g, lower_bound_r = lower_bound
    upper_bound_b, upper_bound_g, upper_bound_r = upper_bound

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Extract the BGR values for the current pixel
            b, g, r = image[y, x]

            # Check if the pixel values are within the specified range for each channel
            if lower_bound_b <= b <= upper_bound_b and \
               lower_bound_g <= g <= upper_bound_g and \
               lower_bound_r <= r <= upper_bound_r:
                mask[y, x] = 255  # Set to 255 if within range

    return mask


def bitwise_and(src: np.array, mask):
    result = np.zeros_like(src)
    result[mask == 255] = src[mask == 255]

    return result
