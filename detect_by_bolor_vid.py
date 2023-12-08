import cv2
import numpy as np
from cv1 import detection, tools


def detect_objects_by_color_rt(target_color_lower, target_color_upper):
    # Define the lower and upper bounds of the target color in HSV
    cap = cv2.VideoCapture(0)
    while True:
        # Capture the current frame
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)
        # Convert the frame from BGR to HSV color space
        hsv_frame = tools.bgr2hsv(frame)
        upper_bound = np.array(target_color_upper)
        lower_bound = np.array(target_color_lower)

        # Create a binary mask where pixels within the color range are white and others are black
        color_mask, contour = detection.in_range_detect(
            hsv_frame, lower_bound, upper_bound
        )
        original = tools.bitwise_and(frame, mask=color_mask)

        color_free = cv2.bitwise_not(color_mask)
        # if cloak is not present show the current image
        result_frame_2 = cv2.bitwise_and(frame, frame, mask=color_free)

        # Draw a rectangle around the detected object
        final = detection.draw_contours(original, contour, color=(0, 0, 255))

        cv2.imshow("color detection", final + result_frame_2)
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


# hadak ta3 draw contour mahouch rsem bien in a video
lower_red = np.array([0, 120, 70])
upper_red = np.array([180, 255, 255])
detect_objects_by_color_rt(lower_red, upper_red)
