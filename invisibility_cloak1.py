import cv2
import numpy as np
from cv1 import detection, tools, filters


def invisibility_cloak(lower_red, upper_red):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Capture the background

    _, background = cap.read()
    cv2.flip(background, 1, background)

    while True:
        # Capture the current frame
        ret, frame = cap.read()
        cv2.flip(frame, 1)
        # Convert the frame from BGR to HSV color space
        # hsv_frame = cv2.bgr2hsv(frame)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Create binary masks for the two color ranges
        color_mask = tools.in_range(
            hsv_frame, lower_red, upper_red
        )
        # removing noise from binary image , remove small bright spots (white regions)
        # color_mask = filters.opening(color_mask, kernel_size=3, iterations=1)
        # enlarge the boundaries of regions of foreground pixels (white regions)
        # color_mask = filters.dilate(color_mask, kernel_size=3, iterations=1)

        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=10)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

        # Extract the cloak color from the frame

        cloak_color = tools.bitwise_and(frame, mask=color_mask)
        # Extract the part of the background corresponding to the cloak
        background_part = tools.bitwise_and(background, mask=color_mask)
        # Replace the cloak color with the background part
        result_frame = tools.add_weighted(cloak_color, 1, background_part, 1, 0)
        # detecting things which are not red
        color_free = tools.bitwise_not(color_mask)
        # if cloak is not present show the current image
        result_frame_2 = tools.bitwise_and(frame, mask=color_free)
        # Display the result in real-time
        cv2.imshow("Invisibility Cloak", result_frame + result_frame_2)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


# Define the color range for the cloak (red color )
lower_red = np.array([0, 120, 70])
upper_red = np.array([180, 255, 255])

invisibility_cloak(lower_red, upper_red)
