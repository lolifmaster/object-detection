import cv2
import numpy as np
from cv1 import detection, tools


def detect_objects_by_color_real_time(
    *,
    lower=None,
    upper=None,
):
    """
    Detects objects in a video stream by color
    :param lower: the lower bound for the color detection
    :param upper: the upper bound for the color detection
    :return: None
    """
    if upper is None:
        upper = [180, 255, 255]
    if lower is None:
        lower = [0, 120, 70]

    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    while True:
        # Capture the current frame
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)
        # Convert the frame from BGR to HSV color space
        hsv_frame = tools.bgr2hsv(frame)
        upper_bound = np.array(upper)
        lower_bound = np.array(lower)

        # Create a binary mask where pixels within the color range are white and others are black
        _, contour = detection.in_range_detect(hsv_frame, lower_bound, upper_bound)

        if contour:
            cv2.rectangle(
                frame,
                (contour[0], contour[1]),
                (contour[2], contour[3]),
                (0, 255, 0),
                2,
            )

        cv2.imshow("color detection", frame)
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_objects_by_color_real_time()
