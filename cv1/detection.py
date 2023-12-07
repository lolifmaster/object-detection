from cv1.tools import range


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
