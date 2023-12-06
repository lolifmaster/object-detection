import cv2
import numpy as np
from cv1.tools import bgr2hsv


def detect_objects_by_color(image_path, target_color_lower, target_color_upper):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to HSV color space
    # hsv_image = bgr2hsv(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print(hsv_image)
    # Define the lower and upper bounds of the target color in HSV
    lower_bound = np.array(target_color_lower)
    upper_bound = np.array(target_color_upper)

    # Create a binary mask where pixels within the color range are white and others are black
    color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=color_mask)

    # Display the original image and the result
    cv2.imshow("Original Image", image)
    cv2.imshow("Detected Objects", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Specify the path to your image and the target color range in HSV
image_path = 'data/orange.png'
target_color_lower = [5, 50, 50]
target_color_upper = [15, 255, 255]

detect_objects_by_color(image_path, target_color_lower, target_color_upper)
