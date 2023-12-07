import cv2
import cv1.filters as filters
from cv1 import Shape, tools

image = cv2.imread('data/ppp.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError('Image not found')

# mean = filters.mean(image, (7, 7))
# median = filters.median(image, (7, 7))
# gaussian = filters.gaussian(image, (7, 7), 1)
# sharpen = filters.sharpen(image)
# laplacian = filters.laplacian(image)
# emboss = filters.emboss(image)
# bilateral = filters.bilateral(image, (7, 7), 75, 75)
# cv2_bilateral = cv2.bilateralFilter(image, 7, 75, 75)
black_image = tools.threshold(image, 127, 255)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
erode = filters.erode(black_image, 5, iterations=5, kernel_shape=Shape.CROSS)
dilate = filters.dilate(black_image, 5, iterations=5, kernel_shape=Shape.CROSS)
cv2_erode = cv2.erode(black_image, kernel, iterations=5)
cv2_dilate = cv2.dilate(black_image, kernel, iterations=5)
opening = filters.opening(black_image, 5, iterations=5, kernel_shape=Shape.CROSS)
closing = filters.closing(black_image, 5, iterations=5, kernel_shape=Shape.CROSS)
cv2_opening = cv2.morphologyEx(black_image, cv2.MORPH_OPEN, kernel, iterations=5)
cv2_closing = cv2.morphologyEx(black_image, cv2.MORPH_CLOSE, kernel, iterations=5)

# show the results
# cv2.imshow('original', image)
# cv2.imshow('mean', mean)
# cv2.imshow('median', median)
# cv2.imshow('gaussian', gaussian)
# cv2.imshow('sharpen', sharpen)
# cv2.imshow('laplacian', laplacian)
# cv2.imshow('emboss', emboss)
# cv2.imshow('bilateral', bilateral)
# cv2.imshow('cv2_bilateral', cv2_bilateral)
cv2.imshow('erode', erode)
cv2.imshow('dilate', dilate)
cv2.imshow('cv2_erode', cv2_erode)
cv2.imshow('cv2_dilate', cv2_dilate)
cv2.imshow('opening', opening)
cv2.imshow('closing', closing)
cv2.imshow('cv2_opening', cv2_opening)
cv2.imshow('cv2_closing', cv2_closing)

cv2.waitKey(0)
cv2.destroyAllWindows()
