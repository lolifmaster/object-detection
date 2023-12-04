import numpy as np
import cv2
from cv1.detection import in_range, find_contours

# Define color range
LO = np.array([100, 30, 120])
HI = np.array([130, 50, 140])


def detect_inrange(image, surface_min, surface_max):
    points = []

    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply color range filter
    mask = in_range(image, LO, HI)

    # Apply morphological operations on the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)

    # Find contours in the mask
    contours = find_contours(mask)

    # Sort contours by area in descending order
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if surface_min < area < surface_max:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            points.append(np.array([int(x), int(y), int(radius), int(area)]))

    return image, mask, points


# Open a video capture object
video_cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = video_cap.read()

    # Flip the frame horizontally
    cv2.flip(frame, 1, frame)

    # Detect objects in the frame
    image, mask, points = detect_inrange(frame, 500, 5000)

    # Draw a circle on the image for visualization
    cv2.circle(image, (100, 100), 20, (0, 255, 0), 5)
    print(image[100, 100])

    # Draw detected points on the original frame
    if len(points) > 0:
        cv2.circle(frame, (points[0][0], points[0][1]), points[0][2], (0, 0, 255), 2)
        cv2.putText(frame, str(points[0][3]), (points[0][0], points[0][1]),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the mask
    if mask is not None:
        cv2.imshow("mask", mask)

    # Display the original frame
    cv2.imshow('image', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release the video capture object
video_cap.release()
cv2.destroyAllWindows()
