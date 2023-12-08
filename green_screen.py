import cv2
import numpy as np
from cv1 import detection, tools


def green_screen_image(img, background_img, lower_green, upper_green):
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


def green_screen_realtime(lower_green, upper_green, *, background_img):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

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
        background = cv2.imread(background_img)
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
        background = tools.bitwise_and(background, mask=tools.bitwise_not(color_mask))
        foreground = tools.add_weighted(foreground, 1, background, 1, 0)

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
upper_bound = np.array([180, 255, 255])

green_screen_realtime(lower_bound, upper_bound, background_img="data/orange.png")
# green_screen_image("data/ppp.png", "data/orange.png", lower_bound, upper_bound)
