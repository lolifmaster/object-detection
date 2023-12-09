import cv2
import numpy as np
from cv1 import detection, tools


# from cv1.tools import range


def detect_objects_by_color(image, target_color_lower, target_color_upper):
    # Read the image
    image = cv2.imread(image)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array(target_color_lower)
    upper_bound = np.array(target_color_upper)

    _, contour = detection.in_range_detect(hsv_image, lower_bound, upper_bound)
    final = image.copy()
    if contour:
        cv2.rectangle(
            final, (contour[0], contour[1]), (contour[2], contour[3]), (0, 255, 0), 2
        )
    cv2.imshow("Original Image", image)
    cv2.imshow("Detected Objects", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_objects_by_color(
        image="data/ppp.jpg",
        target_color_lower=[0, 120, 70],
        target_color_upper=[10, 255, 255],
    )
