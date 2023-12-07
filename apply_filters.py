import cv2
import cv1.filters as filters
import numpy as np

image = cv2.imread('data/ppp.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError('Image not found')

# mean = filters.mean(image, (7, 7))
# median = filters.median(image, (7, 7))
# gaussian = filters.gaussian(image, (7, 7), 1)
# sharpen = filters.sharpen(image)
# laplacian = filters.laplacian(image)
# emboss = filters.emboss(image)
bilateral = filters.bilateral(image, (7, 7), 75, 75)
cv2_bilateral = cv2.bilateralFilter(image, 7, 75, 75)
erode = filters.erode(image, (5, 5), iterations=5)
dilate = filters.dilate(image, (5, 5), iterations=5)
cv2_erode = cv2.erode(image, np.ones((5, 5), dtype=np.uint8), iterations=5)
cv2_dilate = cv2.dilate(image, np.ones((5, 5), dtype=np.uint8), iterations=5)

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

cv2.waitKey(0)
cv2.destroyAllWindows()
