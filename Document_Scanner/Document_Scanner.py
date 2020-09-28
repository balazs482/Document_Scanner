import numpy as np
import cv2

IMAGE_PATH = 'test samples/sample_1.jpg'
THRESHOLD_VALUE = 127

# get image with blur and canny
originalImage = cv2.imread(IMAGE_PATH, -1)
img =  cv2.Canny(cv2.GaussianBlur(originalImage, (5, 5), 1.4), 90, 360) # default: lowThreshold = 100; highTreshold = 200

# find contours
ret, thresh = cv2.threshold(img, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# plot image
cv2.drawContours(originalImage, contours, -1, (0, 255, 0), 3)
cv2.imshow('Frame', originalImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
