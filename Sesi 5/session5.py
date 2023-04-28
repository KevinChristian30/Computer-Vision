import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('chessboard.jpg')
igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

igray = np.float32(igray)

def showResult(source, cmap = None):
  plt.imshow(source, cmap=cmap)
  plt.show()

harris = cv2.cornerHarris(igray, 2, 5, 0.04)

without_subpix = image.copy()
without_subpix[harris > 0.01 * harris.max()] = [0, 255, 0]

# showResult(without_subpix, 'gray')

_, thresh = cv2.threshold(harris, 0.01 * harris.max(), 255, 0)
thresh = np.uint8(thresh)

_, _, _, centroids = cv2.connectedComponentsWithStats(thresh)
centroids = np.float32(centroids)

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.0001)
enhanced_criteria = cv2.cornerSubPix(igray, centroids, (2, 2), (-1, -1), criteria)

enhanced_criteria = np.uint16(enhanced_criteria)

with_subpix = image.copy()

for i in enhanced_criteria:
  x, y = i[:2]
  with_subpix[y, x] = [255, 0, 0]

showResult(with_subpix, 'gray')
