import numpy as np


def calculate_center(contours):
    if not contours:
        return None  # No contours found

    upper_x, upper_y, lower_x, lower_y = contours

    center_x = (upper_x + lower_x) // 2
    center_y = (upper_y + lower_y) // 2

    return center_x, center_y


def in_range_detect(image, lower_bound, upper_bound):
    """
    performs a color detection in the specified range and returns a mask
    and the upper and lower coordinates of the detected object


       :param image: the image to be processed
       :param lower_bound: the lower bound for the color detection
       :param upper_bound: the upper bound for the color detection

       :return: a mask with the detected object and the upper and lower coordinates of the detected object

    """
    if len(image.shape) == 2:
        raise ValueError("Input image must be in HSV format (3 dimensions)")

    # Get the height and width of the image
    height, width, _ = image.shape

    # Initialize an output mask with zeros
    mask = np.zeros((height, width), dtype=np.uint8)

    # Extract lower and upper bounds for each channel
    lower_bound_b, lower_bound_g, lower_bound_r = lower_bound
    upper_bound_b, upper_bound_g, upper_bound_r = upper_bound

    points = []
    for y in range(height):
        for x in range(width):
            # Extract the HSV values for the current pixel
            h, s, v = image[y, x]

            # Check if the pixel values are within the specified range for each channel
            if (
                    lower_bound_b <= h <= upper_bound_b
                    and lower_bound_g <= s <= upper_bound_g
                    and lower_bound_r <= v <= upper_bound_r
            ):
                mask[y, x] = 255  # Set to 255 if within range

                # add the coordinates to the list of points
                points.append((x, y))

    # get the upper and lower coordinates
    if points:
        lower_x = min(points, key=lambda p: p[0])[0]
        lower_y = min(points, key=lambda p: p[1])[1]
        upper_x = max(points, key=lambda p: p[0])[0]
        upper_y = max(points, key=lambda p: p[1])[1]
        cords = (upper_x, upper_y, lower_x, lower_y)
    else:
        cords = None

    return mask, cords
