import cv2
import numpy as np
from cv1 import detection, tools
# from cv1.tools import range


def detect_objects_by_color(image, target_color_lower, target_color_upper):
    # Read the image
    image = cv2.imread(image)

    # Convert the image from BGR to HSV color space
    hsv_image = tools.bgr2hsv(image)
    # Define the lower and upper bounds of the target color in HSV
    lower_bound = np.array(target_color_lower)
    upper_bound = np.array(target_color_upper)

    # Create a binary mask where pixels within the color range are white and others are black
    color_mask,contous = detection.in_range(hsv_image, lower_bound, upper_bound)


    original = detection.bitwise_and(image, mask=color_mask)

    # Draw a rectangle around the detected object
    final= detection.draw_contours(original,contous, color=(0 , 0, 255))



    # Display the original image and the result
    cv2.imshow("Original Image", image)
    cv2.imshow("Detected Objects", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Specify the path to your image and the target color range in HSV
image_path = 'data/orange.png'

detect_objects_by_color(image_path, [5, 50, 50], [15, 255, 255])
