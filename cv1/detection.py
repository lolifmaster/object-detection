from cv1.tools import range
import numpy as np


def draw_contours(image, contours, color=(0, 255, 0)):
    """color the rectangle around the detected contour """

    upper_x_contour, upper_y_contour, lower_x_contour, lower_y_contour = contours

    # color the upper and lower rows
    for x in range(upper_x_contour, lower_x_contour):
        image[upper_y_contour, x] = color
        image[lower_y_contour, x] = color

    # color the left and right columns
    for y in range(upper_y_contour, lower_y_contour):
        image[y, upper_x_contour] = color
        image[y, lower_x_contour] = color

    return image


def calculate_center(contours):
    if not contours:
        return None  # No contours found

    upper_x, upper_y, lower_x, lower_y = contours

    center_x = (upper_x + lower_x) // 2
    center_y = (upper_y + lower_y) // 2

    return center_x, center_y


def in_range_detect(image, lower_bound, upper_bound):
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

    lower_x = width
    lower_y = height
    upper_x = 0
    upper_y = 0

    for y in range(height):
        for x in range(width):
            # Extract the BGR values for the current pixel
            h, s, v = image[y, x]

            # Check if the pixel values are within the specified range for each channel
            if lower_bound_b <= h <= upper_bound_b and \
                    lower_bound_g <= s <= upper_bound_g and \
                    lower_bound_r <= v <= upper_bound_r:
                mask[y, x] = 255  # Set to 255 if within range

                # find the upper and lower coordinates of the colored area
                lower_x = lower_x if lower_x < x else x
                lower_y = lower_y if lower_y < y else y
                upper_x = upper_x if upper_x > x else x
                upper_y = upper_y if upper_y > y else y

    return mask, (upper_x, upper_y, lower_x, lower_y)
