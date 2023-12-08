import cv2
import numpy as np
from cv1 import detection, tools 


def bitwise_not(image):
    """
    Perform bitwise NOT operation on the input image.

    Args:
        image (list[list[int]]): The input image represented as a 2D list.

    Returns:
        list[list[int]]: The result of the bitwise NOT operation.
    """
    # Ensure the input image is a list of lists
    if not isinstance(image, list) or not all(isinstance(row, list) for row in image):
        raise ValueError("Input should be a 2D list")

    # Ensure the image has only one channel (grayscale)
    if any(len(row) != len(image[0]) for row in image):
        raise ValueError("Input should be a grayscale image with consistent row lengths")

    # Invert pixel values 
    result = [[255 - pixel for pixel in row] for row in image]

    return result

def add_weighted(img1, alpha, img2, beta, gamma):
    """
    Perform weighted sum of two images.

    Args:
        img1 (numpy.ndarray): The first input image.
        alpha (float): Weight for the first image.
        img2 (numpy.ndarray): The second input image.
        beta (float): Weight for the second image.
        gamma (float): Scalar added to each sum.

    Returns:
        numpy.ndarray: The result of the weighted sum.
    """
    # Ensure input images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape")

    # Create an empty result image
    result = img1.copy()

    # Iterate over each pixel and perform the weighted addition
    for y in range(img1.shape[0]):
        for x in range(img1.shape[1]):
            pixel1 = img1[y, x]
            pixel2 = img2[y, x]

            # Perform the weighted addition for each channel
            new_pixel = tuple(
                max(0, min(int(p1 * alpha + p2 * beta + gamma), 255)) for p1, p2 in zip(pixel1, pixel2)
            )

            # Update the result image with the new pixel value
            result[y, x] = new_pixel

    return result

def invisibility_cloak(lower_red, upper_red):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

    # Wait for a few seconds to allow the webcam to start
    cv2.waitKey(2000)

    # Capture the background
    
    _, background = cap.read()
    cv2.flip(background, 1, background)

    while True:
        # Capture the current frame
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)
        # Convert the frame from BGR to HSV color space
        # hsv_frame = cv2.bgr2hsv(frame)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Create binary masks for the two color ranges
        color_mask, contour1 = detection.in_range_detect(hsv_frame, lower_red, upper_red)
        # # removing noise from binary image , remove small bright spots (white regions) 
        # color_mask = filters.opening(color_mask, kernel_size=3, iterations=10)
        # # enlarge the boundaries of regions of foreground pixels (white regions)
        # color_mask = filters.dilate(color_mask, kernel_size=3, iterations=1)

        # color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations = 10) 
        # color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations = 1) 


        # Extract the cloak color from the frame

        cloak_color = tools.bitwise_and(frame, mask=color_mask)
        # Extract the part of the background corresponding to the cloak
        background_part = tools.bitwise_and(background, mask=color_mask)
        # Replace the cloak color with the background part
        result_frame = cv2.addWeighted(cloak_color, 1, background_part, 1, 0)
        # detecting things which are not red
        color_free = bitwise_not(color_mask)
        # if cloak is not present show the current image
        result_frame_2 = tools.bitwise_and(frame, mask= color_free)
        # Display the result in real-time
        cv2.imshow("Invisibility Cloak", result_frame + result_frame_2)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()



# Define the color range for the cloak (red color )
lower_red = np.array([0, 120, 70])
upper_red = np.array([180, 255, 255])


invisibility_cloak(lower_red, upper_red)



