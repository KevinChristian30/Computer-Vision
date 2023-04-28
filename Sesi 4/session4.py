import cv2
import numpy as np
import matplotlib.pyplot as plt

def showResult(nrow = None, ncol = None, res_stack = None):
  plt.figure(figsize=(12, 12))

  for i, (label, image) in enumerate(res_stack):
    plt.subplot(nrow, ncol, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(label)
    plt.axis('off')
  
  plt.show()

image = cv2.imread('fruits.jpg')
igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

laplace8u = cv2.Laplacian(igray, cv2.CV_8U)
laplace16s = cv2.Laplacian(igray, cv2.CV_16S)
laplace32f = cv2.Laplacian(igray, cv2.CV_32F)
laplace64f = cv2.Laplacian(igray, cv2.CV_64F)

laplace_labels = ['8U', '16S', '32F', '64F']
laplace_images = [laplace8u, laplace16s, laplace32f, laplace64f]

# showResult(2, 2, zip(laplace_labels, laplace_images))

k_size = 3
sobel_x = cv2.Sobel(igray, cv2.CV_32F, 1, 0, k_size)
sobel_y = cv2.Sobel(igray, cv2.CV_32F, 0, 1, k_size)

sobel_labels = ['Sobel X', 'Sobel Y']
sobel_images = [sobel_x, sobel_y]

# showResult(1, 2, zip(sobel_labels, sobel_images))

merged_sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
merged_sobel *= 255 / merged_sobel.max()

# showResult(1, 1, zip(['Merged Sobel'], [merged_sobel]))

canny50100 = cv2.Canny(igray, 50, 100)
canny50150 = cv2.Canny(igray, 50, 150)
canny75150 = cv2.Canny(igray, 75, 150)
canny75225 = cv2.Canny(igray, 75, 225)

canny_labels = ['50 100', '50 150', '75 150', '75 225']
canny_images = [canny50100, canny50150, canny75150, canny75225]
showResult(2, 2, zip(canny_labels, canny_images))
