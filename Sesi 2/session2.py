import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('model.jpg')
igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def showResult(label = None, image = None, cmap = None):
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.hist(image.flat, bins=256, range=(0, 256))
  plt.title(label)
  plt.xlabel('Intensity Value')
  plt.ylabel('Intensity Quantity')
  plt.subplot(1, 2, 2)
  plt.imshow(image, cmap=cmap)
  plt.axis('off')
  plt.show()

normal_image = igray.copy()
# showResult('Normal Image', normal_image, 'gray')

normal_equ_hist = cv2.equalizeHist(igray)
# showResult('Normal Equalized Histogram Image', normal_equ_hist, 'gray')

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_equ_hist = clahe.apply(igray.copy())
# showResult('CLAHE Histogram Image', normal_equ_hist, 'gray')

hist_labels = ['normal', 'nequ', 'cequ']
hist_images = [normal_image, normal_equ_hist, clahe_equ_hist]
# for i, (label, image) in enumerate(zip(hist_labels, hist_images)):
#   plt.subplot(3, 1, i + 1)
#   plt.hist(image.flat, bins=256, range=(0, 256))
#   plt.title(label)
#   plt.xlabel('Intensity Value')
#   plt.ylabel('Intensity Quantity')

# plt.show()

for i, (label, image) in enumerate(zip(hist_labels, hist_images)):
  plt.subplot(1, 3, i + 1)
  plt.imshow(image, cmap='gray')
  plt.title(label)
  plt.axis('off')

plt.show()