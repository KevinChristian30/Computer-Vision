import cv2
import os
from scipy.spatial.distance import euclidean

# Train
image_dir = 'images/train_image'
features = []

for filename in os.listdir(image_dir):
  image_name = filename.split('.')[0]
  image = cv2.imread(f'{image_dir}/{filename}')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
  norm = cv2.normalize(hist, None)

  flat = norm.flatten()

  features.append((image_name, flat))

# Test
test_image = cv2.imread('images/test_image/tart-02.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

hist = cv2.calcHist([test_image], [0, 1, 2],  None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
norm = cv2.normalize(hist, None)
flat = norm.flatten()

result = []

for name, hist in features:
  distance = euclidean(hist, flat)
  result.append((distance, name))

result = sorted(result)

for i, j in result:
  print(f'{i} : {j}')