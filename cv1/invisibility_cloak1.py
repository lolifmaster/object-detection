import cv2
import numpy as np
from cv1 import tools, detection


def invisibility_cloak(*, lower_red, upper_red, background_img=None):
    if not lower_red:
        lower_red = np.array([0, 120, 70])
    if not upper_red:
        upper_red = np.array([10, 255, 255])
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Capture the background

    if background_img is not None:
        background_img = cv2.imread(background_img)
        background = cv2.resize(background_img, (320, 240))
    else:
        _, background = cap.read()
        cv2.flip(background, 1, background)

    while True:
        # Capture the current frame
        ret, frame = cap.read()
        cv2.flip(frame, 1)
        # Convert the frame from BGR to HSV color space
        # hsv_frame = cv2.bgr2hsv(frame)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask, contour = detection.in_range_detect(hsv_frame, lower_red, upper_red)

        if contour:
            upper_x, upper_y, lower_x, lower_y = contour
            frame[lower_y:upper_y, lower_x:upper_x] = background[
                lower_y:upper_y, lower_x:upper_x
            ]

        cv2.imshow("Invisibility Cloak", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    invisibility_cloak(lower_red=[0, 120, 70], upper_red=[10, 255, 255])
