import cv2
import numpy as np
from cv1 import tools


def green_screen_image(*, img, background_img, lower_green=None, upper_green=None):
    if not lower_green:
        lower_green = np.array([0, 120, 70])
    if not upper_green:
        upper_green = np.array([10, 255, 255])
    # Load the image
    image = cv2.imread(img)
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Create binary masks for the color range
    color_mask = tools.in_range(hsv_image, lower_green, upper_green)
    # Extract the foreground (object) from the frame
    foreground = tools.bitwise_and(image, mask=color_mask)

    # put a background image
    background = cv2.imread(background_img)
    background = cv2.resize(background, (image.shape[1], image.shape[0]))
    background = tools.bitwise_and(background, mask=tools.bitwise_not(color_mask))
    foreground = tools.add_weighted(foreground, 1, background, 1, 0)
    # Display the result
    cv2.imshow("Green Screen", foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def green_screen_realtime(*, lower_green=None, upper_green=None, background_img=None):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not lower_green:
        lower_green = np.array([0, 120, 70])
    if not upper_green:
        upper_green = np.array([10, 255, 255])
    if background_img is None:
        _, background = cap.read()
        cv2.flip(background, 1, background)
    else:
        background = cv2.imread(background_img)
        background = cv2.resize(background, (320, 240))

    while True:
        # Capture the current frame
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)
        # Convert the frame from BGR to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Create binary masks for the color range
        color_mask = tools.in_range(hsv_frame, lower_green, upper_green)
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=10
        )
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1
        )
        # Extract the foreground (object) from the frame
        foreground = tools.bitwise_and(frame, mask=color_mask)

        # put a background image
        current_background = tools.bitwise_and(
            background, mask=tools.bitwise_not(color_mask)
        )
        foreground = tools.add_weighted(foreground, 1, current_background, 1, 0)

        # Display the result in real-time
        cv2.imshow("Green Screen ", foreground)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
