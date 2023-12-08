# import cv2
# import numpy as np
# from cv1 import detection, tools






# def green_screen(lower_red, upper_red):
#     # Initialize the webcam
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

#     # Wait for a few seconds to allow the webcam to start
#     cv2.waitKey(2000)

#     # Capture the background
#     background = cv2.imread("data/ppp.png")

#     while True:
#         ret, frame = cap.read()
#         cv2.flip(frame, 1, frame)
#         hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         # Create binary masks for the two color ranges
#         color_mask, contour1 = detection.in_range_detect(hsv_frame, lower_red, upper_red)

#         cloak_color = cv2.bitwise_and(frame,frame, mask=color_mask)
#         # Extract the part of the background corresponding to the cloak
#         background_part = cv2.bitwise_not(color_mask)
#         # Replace the cloak color with the background part
#         result_frame = cv2.addWeighted(cloak_color, 1, background_part, 1, 0)
#         # detecting things which are not red
#         color_free = cv2.bitwise_not(color_mask)
#         # if cloak is not present show the current image
#         result_frame_2 = cv2.bitwise_and(frame,frame, mask= color_free)
#         # Display the result in real-time
#         cv2.imshow("Invisibility Cloak", result_frame + result_frame_2)

#         # Break the loop if the 'q' key is pressed
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break



# # Call the green_screen function
# lower_red = np.array([0, 120, 70])
# upper_red = np.array([180, 255, 255])

# green_screen(lower_red, upper_red)

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
        color_mask, contour = detection.in_range_detect(hsv_frame, lower_bound, upper_bound)
        # Extract the foreground (object) from the frame
        foreground = tools.bitwise_and(frame, mask=color_mask)
        # Display the result in real-time
        cv2.imshow("Green Screen ", foreground)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Specify the target color range in HSV
lower_bound = np.array([0, 120, 70])
upper_bound = np.array([10, 255, 255])

green_screen_realtime(lower_bound, upper_bound)
