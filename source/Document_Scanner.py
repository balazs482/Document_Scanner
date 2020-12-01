import numpy as np
import cv2
import itertools

import constants

# get image with blur and canny
originalImage = cv2.imread(constants.IMAGE_PATH, -1)
img =  cv2.Canny(cv2.GaussianBlur(originalImage, (5, 5), constants.BLUR_VALUE), constants.CANNY_LOWTHRESHOLD, constants.CANNY_HIGHTRESHOLD)
HEIGHT, WIDTH = img.shape

# find contours
ret, thresh = cv2.threshold(img, constants.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# contour approximation
contours = sorted(contours, key = cv2.contourArea, reverse = True)
for i in contours:
    elip =  cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, constants.ELLIPSE_COEFFICIENT * elip, True)
    if len(approx) == 4:
        doc = approx
        break
doc = doc.reshape((4,2))

# checking if four corners are recognized and far apart
if len(doc) != 4:
    print('ERROR: Incorrect number of corners detected.')

else:
    # calculating minimum distance between points
    minDistance = min(HEIGHT, WIDTH)
    for pair in itertools.combinations(doc, r = 2):
        distance = np.linalg.norm(pair[0] - pair[1])
        if (distance < minDistance):
            minDistance = distance
    if minDistance < HEIGHT / constants.CORNER_DISTANCE_RATIO:
        # error
        print('ERROR: Corners are too close together')
    else:
        # assign corners
        topCorners, bottomCorners = np.array_split(doc[np.argsort(doc[:, 1])], 2)     
        topCorners = topCorners[np.argsort(topCorners[:, 0])]
        bottomCorners = bottomCorners[np.argsort(bottomCorners[:, 0])]
        # getting larger vertical and horizontal distances
        def dist(p1, p2): return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        maxWidth = max(int(dist(topCorners[0], topCorners[1])), int(dist(bottomCorners[0], bottomCorners[1])))
        maxHeight = max(int(dist(topCorners[0], bottomCorners[0])), int(dist(topCorners[1], bottomCorners[1])))

        # creating destination array
        destinationPoints = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [0, maxHeight - 1],
        [maxWidth - 1, maxHeight - 1]
        ], dtype = "float32")

        # transform with matrix
        corners = np.array(np.concatenate((topCorners, bottomCorners), axis = 0), np.float32)
        img = cv2.warpPerspective(originalImage, cv2.getPerspectiveTransform(corners, destinationPoints), (maxWidth, maxHeight))

        # plotting image
        cv2.imshow('Frame', cv2.resize(img, (0, 0), fx = constants.PLOTTING_RATIO, fy = constants.PLOTTING_RATIO))

cv2.waitKey(0)
cv2.destroyAllWindows()
