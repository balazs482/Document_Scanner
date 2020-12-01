import cv2
import Document_Scanner
import constants

img = Document_Scanner.warp('./test samples/sample_1.jpg')
cv2.imshow('Frame', cv2.resize(img, (0, 0), fx = constants.PLOTTING_RATIO, fy = constants.PLOTTING_RATIO))
cv2.waitKey(0)
cv2.destroyAllWindows()
