import cv2
import numpy as np
from cv1 import detection, tools


def detect_objects_by_color_real_time(
    *,
    lower=None,
    upper=None,
):
    """
    Detects objects in a video stream by color.

    Args:
    - lower: The lower bound for color detection.
    - upper: The upper bound for color detection.

    Returns:
    - None
    """
    # Set default upper and lower bounds if not provided
    if upper is None:
        upper = [180, 255, 255]  # Default upper bound for HSV color space
    if lower is None:
        lower = [0, 120, 70]  # Default lower bound for HSV color space

    # Open the webcam (camera index 0)
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)  # Set the width of the frame
    cap.set(4, 240)  # Set the height of the frame

    while True:
        # Capture the current frame from the webcam
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)  # Flip the frame horizontally

        # Convert the frame from BGR to HSV color space
        hsv_frame = tools.bgr2hsv(frame)
        upper_bound = np.array(upper)
        lower_bound = np.array(lower)

        # Create a binary mask where pixels within the color range are white and others are black
        _, contour = detection.in_range_detect(hsv_frame, lower_bound, upper_bound)

        # If an object is detected, draw a rectangle around it
        if contour:
            cv2.rectangle(
                frame,
                (contour[0], contour[1]),
                (contour[2], contour[3]),
                (0, 255, 0),  # Green color for the rectangle
                2,  # Thickness of the rectangle
            )

        # Display the frame with color detection
        cv2.imshow("color detection", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
