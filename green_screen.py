import cv2
import numpy as np
from cv1 import detection, tools


def green_screen_realtime(lower_bound, upper_bound):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

    # Wait for a few seconds to allow the webcam to start
    cv2.waitKey(2000)
    # Define a kernel for morphological operations
    while True:
        # Capture the current frame
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)
        # Convert the frame from BGR to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Create binary masks for the color range
        color_mask, contour = detection.in_range_detect(
            hsv_frame, lower_bound, upper_bound
        )
        # Extract the foreground (object) from the frame
        foreground = tools.bitwise_and(frame, mask=color_mask)
        # Display the result in real-time
        cv2.imshow("Green Screen ", foreground)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


# Specify the target color range in HSV
lower_bound = np.array([0, 120, 70])
upper_bound = np.array([10, 255, 255])

green_screen_realtime(lower_bound, upper_bound)
