import cv2
import numpy as np


def bilateral_filter(image, d, sigma_color, sigma_space):
    rows, cols, _ = image.shape
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            pixel_value = 0.0
            normalization_factor = 0.0
            for x in range(max(0, i - d), min(rows, i + d + 1)):
                for y in range(max(0, j - d), min(cols, j + d + 1)):
                    spatial_diff = np.sqrt((x - i) ** 2 + (y - j) ** 2)
                    intensity_diff = np.linalg.norm(image[x, y] - image[i, j])
                    weight = np.exp(
                        -(spatial_diff**2) / (2 * sigma_space**2)
                        - intensity_diff**2 / (2 * sigma_color**2)
                    )

                    pixel_value += weight * image[x, y]
                    normalization_factor += weight

            result[i, j] = pixel_value / normalization_factor

    return result.astype(np.uint8)


# Read the image
image = cv2.imread("data/ppp.png")

# Convert image to float32 for bilateral filter
image_float32 = image.astype(np.float32)
# Apply Bilateral Filter (You might need to adjust the parameters)
bilateral_filtered_image = bilateral_filter(
    image_float32, d=9, sigma_color=75, sigma_space=75
)
# Apply Bilateral Filter
bilateral_filtered_imageO = cv2.bilateralFilter(
    image_float32, d=9, sigmaColor=75, sigmaSpace=75
)
# # Display the original and filtered images
cv2.imshow("Original Image", image)
cv2.imshow("mine", bilateral_filtered_image)
cv2.imshow("cv2", bilateral_filtered_imageO)
cv2.waitKey(0)
cv2.destroyAllWindows()
