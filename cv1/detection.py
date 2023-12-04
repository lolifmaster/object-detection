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


def find_contours(binary_image):
    contours = []

    # Find all white pixels in the binary image
    white_pixels = np.column_stack(np.where(binary_image > 0))

    # Iterate over each white pixel

    for y, x in white_pixels:
        # Check if the pixel has already been visited
        if binary_image[y, x] == 0:
            continue

        # Initialize a contour
        contour = []

        # Add the current pixel to the contour
        contour.append((y, x))

        # Mark the current pixel as visited
        binary_image[y, x] = 0

        # Initialize the current direction
        direction = 0

        # Iterate over the 7 possible directions
        for i in range(7):
            # Compute the next direction
            direction = (direction + 1) % 8

            # Compute the coordinates of the next pixel
            if direction == 0:
                next_y, next_x = y - 1, x
            elif direction == 1:
                next_y, next_x = y - 1, x + 1
            elif direction == 2:
                next_y, next_x = y, x + 1
            elif direction == 3:
                next_y, next_x = y + 1, x + 1
            elif direction == 4:
                next_y, next_x = y + 1, x
            elif direction == 5:
                next_y, next_x = y + 1, x - 1
            elif direction == 6:
                next_y, next_x = y, x - 1
            elif direction == 7:
                next_y, next_x = y - 1, x - 1

            # Check if the next pixel is within the image
            if next_y < 0 or next_y >= binary_image.shape[0] or \
               next_x < 0 or next_x >= binary_image.shape[1]:
                continue

            # Check if the next pixel is white
            if binary_image[next_y, next_x] > 0:
                # Add the next pixel to the contour
                contour.append((next_y, next_x))

                # Mark the next pixel as visited
                binary_image[next_y, next_x] = 0

                # Break the loop
                break

        # Check if the contour is valid
        if len(contour) > 2:
            contours.append(contour)

    return contours
