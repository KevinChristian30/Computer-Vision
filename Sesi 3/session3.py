import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('lena.jpg')
height, width = image.shape[:2]

def showResult(nrow = None, ncol = None, res_stack = None):
  plt.figure(figsize=(12, 12))

  for i, (label, image) in enumerate(res_stack):
    plt.subplot(nrow, ncol, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(label)
    plt.axis('off')
  
  plt.show()

gray_ocv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_avg = np.dot(image, [0.33, 0.33, 0.33])

image_b, image_g, image_r = image[:, :, 0], image[:, :, 1], image[:, :, 2]

max_cha = max(np.max(image_b), np.max(image_g), np.max(image_r))
min_cha = min(np.min(image_b), np.min(image_g), np.min(image_r))

gray_lig = np.dot(image, [(max_cha + min_cha) / 2, (max_cha + min_cha) / 2, (max_cha + min_cha) / 2])
gray_lum = np.dot(image, [0.07, 0.71, 0.21])
gray_wag = np.dot(image, [0.114, 0.507, 0.299])

gray_labels = ['ocv', 'avg', 'lig', 'lum', 'wag']
gray_images = [gray_ocv, gray_avg, gray_lig, gray_lum, gray_wag]

# showResult(3, 2, zip(gray_labels, gray_images))
thresh = 100
thresh_image = gray_ocv.copy()

for i in range(height):
  for j in range(width):
    if thresh_image[i, j] > thresh:
      thresh_image[i, j] = 255
    else:
      thresh_image[i, j] = 0

# showResult(1, 1, zip(['Manual Threshold'], [thresh_image]))

_, bin_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_BINARY_INV)
_, binv_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_BINARY)
_, mask_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_MASK)
_, otsu_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_OTSU)
_, toz_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_TOZERO)
_, tozinv_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_TOZERO_INV)
_, tri_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_TRIANGLE)
_, trunc_thresh = cv2.threshold(gray_ocv, 100, 255, cv2.THRESH_TRUNC)

thresh_labels = ['man', 'bin', 'binverse', 'mask',
                 'otsu', 'tozero', 'tozeroinv', 'triangle', 'trunc']
thresh_images = [thresh_image, bin_thresh, binv_thresh, mask_thresh,
                 otsu_thresh, toz_thresh, tozinv_thresh, tri_thresh, trunc_thresh]

showResult(3, 3, zip(thresh_labels, thresh_images))

# Filtering
def manual_mean_filter(source=None, ksize=None):
    np_source = np.array(source)
    for i in range(height - ksize - 1):
        for j in range(width - ksize - 1):
            matrix = np.array(
                np_source[i: (i + ksize), j: (j + ksize)]).flatten()
            mean = np.mean(matrix)
            np_source[i + ksize//2, j + ksize//2] = mean
    return np_source


def manual_median_filter(source=None, ksize=None):
    np_source = np.array(source)
    for i in range(height - ksize - 1):
        for j in range(width - ksize - 1):
            matrix = np.array(
                np_source[i: (i + ksize), j: (j + ksize)]).flatten()
            median = np.median(matrix)
            np_source[i + ksize//2, j + ksize//2] = median
    return np_source


# B G R
b, g, r = cv2.split(image)
mean_b = manual_mean_filter(b, 3)
mean_g = manual_mean_filter(g, 3)
mean_r = manual_mean_filter(r, 3)

median_b = manual_median_filter(b, 3)
median_g = manual_median_filter(g, 3)
median_r = manual_median_filter(r, 3)

merged_mean = cv2.merge((mean_b, mean_g, mean_r))
merged_median = cv2.merge((median_b, median_g, median_r))

blurred_image = gray_ocv.copy()

blur = cv2.blur(blurred_image, (3, 3))
median_blur = cv2.medianBlur(blurred_image, 3)
gauss_blur = cv2.GaussianBlur(blurred_image, (3, 3), 2.0)
bilateral_blur = cv2.bilateralFilter(blurred_image, 3, 150, 150)

blur_labels = ['blur', 'median blur', 'gaussian blur',
               'bilateral blur', 'merged mean', 'merged median']
blur_images = [blur, median_blur, gauss_blur,
               bilateral_blur, merged_mean, merged_median]

showResult(2, 3, zip(blur_labels, blur_images))