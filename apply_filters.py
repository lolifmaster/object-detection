import cv2
import cv1.filters as filters
from cv1 import Shape, tools

image = cv2.imread("data/ppp.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not found")

# mean = filters.mean(image, (7, 7))
# median = filters.median(image, (7, 7))
# gaussian = filters.gaussian(image, (7, 7), 1)
# sharpen = filters.sharpen(image)
# laplacian = filters.laplacian(image)
# emboss = filters.emboss(image)
sobel = filters.sobel(image)
# edge_detection = filters.edge_detection(image)

# black_image = tools.threshold(image, 127, 255)
# erode = filters.erode(black_image, 5, iterations=5, kernel_shape=Shape.CROSS)
# dilate = filters.dilate(black_image, 5, iterations=5, kernel_shape=Shape.CROSS)
# opening = filters.opening(black_image, 5, iterations=5, kernel_shape=Shape.CROSS)
# closing = filters.closing(black_image, 5, iterations=5, kernel_shape=Shape.CROSS)

# show the results
# cv2.imshow('original', image)
# cv2.imshow('mean', mean)
# cv2.imshow('median', median)
# cv2.imshow('gaussian', gaussian)
# cv2.imshow('sharpen', sharpen)
# cv2.imshow('laplacian', laplacian)
# cv2.imshow('emboss', emboss)
cv2.imshow("sobel", sobel)
# cv2.imshow("erode", erode)
# cv2.imshow("dilate", dilate)
# cv2.imshow("opening", opening)
# cv2.imshow("closing", closing)
# cv2.imshow("edge_detection", edge_detection)

cv2.waitKey(0)
cv2.destroyAllWindows()
