import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import os

# Train
train_path = 'images/train'
train_dir_list = os.listdir(train_path)

image_list = []
image_class_list = []

for idx, train_dir in enumerate(train_dir_list):
  dir_path = os.listdir(f'{train_path}/{train_dir}')
  for image_path in dir_path:
    image_list.append(f'{train_path}/{train_dir}/{image_path}')
    image_class_list.append(idx)

sift = cv2.SIFT_create()

descriptor_list = []

for image_path in image_list:
  _, ds = sift.detectAndCompute(cv2.imread(image_path), None)
  descriptor_list.append(ds)

# Tampung ke stack_ds yang index ke 0
stack_ds = descriptor_list[0]

# Sisanya kita tampung 1 per 1
for ds in descriptor_list[1:]:
  stack_ds = np.vstack((stack_ds, ds))

stack_ds = np.float32(stack_ds)
centroids, _ = kmeans(stack_ds, 100, 1)

image_features = np.zeros((len(image_list), len(centroids)), "float32")

for i in range(0, len(image_list)):
  words, _ = vq(descriptor_list[i], centroids)
  for w in words:
    image_features[i][w] += 1

stdScaler = StandardScaler().fit(image_features)
image_features = stdScaler.transform(image_features)

svc = LinearSVC()
svc.fit(image_features, np.array(image_class_list))

# Test
test_path = 'images/test'
image_list = []

for path in os.listdir(test_path):
  image_list.append(f'{test_path}/{path}')

descriptor_list = []

for image_path in image_list:
  _, ds = sift.detectAndCompute(cv2.imread(image_path), None)
  descriptor_list.append(ds)

# Tampung ke stack_ds yang index ke 0
stack_ds = descriptor_list[0]

# Sisanya kita tampung 1 per 1
for ds in descriptor_list[1:]:
  stack_ds = np.vstack((stack_ds, ds))

stack_ds = np.float32(stack_ds)
centroids, _ = kmeans(stack_ds, 100, 1)

test_features = np.zeros((len(image_list), len(centroids)), "float32")

for i in range(0, len(image_list)):
  words, _ = vq(descriptor_list[i], centroids)
  for w in words:
    test_features[i][w] += 1

test_features = stdScaler.transform(test_features)
result = svc.predict(test_features)

for class_id, image_path in zip(result, image_class_list):
  print(f"{image_path} : {train_dir_list[class_id]}")
