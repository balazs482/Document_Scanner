import numpy as np
import cv2
import itertools

IMAGE_PATH = 'test samples/sample_1.jpg'
BLUR_VALUE = 10
CANNY_LOWTHRESHOLD = 40 # default is 100
CANNY_HIGHTRESHOLD = 300 # default is 200
THRESHOLD_VALUE = 127
ELLIPSE_COEFFICIENT = 0.08
CORNER_DISTANCE_RATIO = 3
# get image with blur and canny
originalImage = cv2.imread(IMAGE_PATH, -1)
img =  cv2.Canny(cv2.GaussianBlur(originalImage, (5, 5), BLUR_VALUE), CANNY_LOWTHRESHOLD, CANNY_HIGHTRESHOLD)
HEIGHT, WIDTH = img.shape

# find contours
ret, thresh = cv2.threshold(img, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# contour approximation
contours = sorted(contours, key = cv2.contourArea, reverse = True)
for i in contours:
	elip =  cv2.arcLength(i, True)
	approx = cv2.approxPolyDP(i, ELLIPSE_COEFFICIENT * elip, True)

	if len(approx) == 4: 
		doc = approx 
		break

# checking if four corners are recognized and far apart
if len(doc) != 4:
    # error
    print('ERROR: Could not recognize 4 corners')
else:
    # calculating minimum distance between points
    minDistance = math.sqrt(HEIGHT ** 2 + WIDTH ** 2)
    for pair in itertools.combinations(doc, r = 2):
        distance = np.linalg.norm(pair[0] - pair[1])
        if (distance < minDistance): 
            minDistance = distance
    if minDistance < HEIGHT / CORNER_DISTANCE_RATIO:
        #error
        print('ERROR: Corners are too close together')
    else:
        # plot image
        cv2.drawContours(originalImage, [doc], -1, (0, 255, 0), 3)
	cv2.imshow('Prepared image', img)
        cv2.imshow('Recognized sheet', originalImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
