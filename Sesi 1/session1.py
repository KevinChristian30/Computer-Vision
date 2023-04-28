import cv2
import numpy as np

def showResult(win_name = None, image = None):
  cv2.imshow(win_name, image)
  cv2.waitKey()
  cv2.destroyAllWindows()

image = cv2.imread('lena.jpg')
image_r = image.copy()
image_g = image.copy()
image_b = image.copy()

# (B, G, R) (0, 1, 2)
image_r[:, :, (0, 1)] = 0
image_g[:, :, (0, 2)] = 0
image_b[:, :, (1, 2)] = 0

# showResult('Lena', image_r)
# showResult('Lena', image_g)
# showResult('Lena', image_b)

image_vstack = np.vstack((image_r, image_g, image_b))
image_hstack = np.hstack((image_r, image_g, image_b))

showResult('', image_vstack)
showResult('', image_hstack)