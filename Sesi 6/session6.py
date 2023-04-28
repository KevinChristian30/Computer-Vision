import cv2
import numpy as np
import matplotlib.pyplot as plt

image_object = cv2.imread('marjan.png')
image_scene = cv2.imread('marjan_banyak.png')

SIFT = cv2.SIFT_create()
ORB = cv2.ORB_create()
AKAZE = cv2.AKAZE_create()

sift_keypoint_object, sift_descriptor_object = SIFT.detectAndCompute(image_object, None)
sift_keypoint_scene, sift_descriptor_scene = SIFT.detectAndCompute(image_scene, None)

orb_keypoint_object, orb_descriptor_object = ORB.detectAndCompute(image_object, None)
orb_keypoint_scene, orb_descriptor_scene = ORB.detectAndCompute(image_scene, None)

akaze_keypoint_object, akaze_descriptor_object = AKAZE.detectAndCompute(image_object, None)
akaze_keypoint_scene, akaze_descriptor_scene = AKAZE.detectAndCompute(image_scene, None)

# Sift dan Akaze perlu diubah ke float32 karena mau cari euclidean distance
sift_descriptor_object = np.float32(sift_descriptor_object)
sift_descriptor_scene = np.float32(sift_descriptor_scene)

akaze_descriptor_object = np.float32(akaze_descriptor_object)
akaze_descriptor_scene = np.float32(akaze_descriptor_scene)

# Orb pake hamming
flann = cv2.FlannBasedMatcher(dict(algorithm = 1), dict(checks = 50))
bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = 2)

sift_match = flann.knnMatch(sift_descriptor_object, sift_descriptor_scene, 2)
akaze_match = flann.knnMatch(akaze_descriptor_object, akaze_descriptor_scene, 2)

orb_match = bfmatcher.match(orb_descriptor_object, orb_descriptor_scene)
orb_match = sorted(orb_match, key = lambda x : x.distance)

def createMasking(mask, match):
  for i, (fm, sm) in enumerate(match):
    if fm.distance < 0.7 * sm.distance:
      mask[i] = [1, 0]
  return mask

sift_matches_mask = [[0, 0] for i in range(0, len(sift_match))]
akaze_matches_mask = [[0, 0] for i in range(0, len(akaze_match))]

sift_matches_mask = createMasking(sift_matches_mask, sift_match)
akaze_matches_mask = createMasking(akaze_matches_mask, akaze_match)

sift_res = cv2.drawMatchesKnn(
  image_object, sift_keypoint_object,
  image_scene, sift_keypoint_scene,
  sift_match, None,
  matchColor = [255, 0, 0],
  singlePointColor = [0, 255, 0],
  matchesMask = sift_matches_mask
)

akaze_res = cv2.drawMatchesKnn(
  image_object, akaze_keypoint_object,
  image_scene, akaze_keypoint_scene,
  akaze_match, None,
  matchColor = [255, 0, 0],
  singlePointColor = [0, 255, 0],
  matchesMask = akaze_matches_mask
)

orb_res = cv2.drawMatches(
  image_object, orb_keypoint_object,
  image_scene, orb_keypoint_scene,
  orb_match[:20], None,
  matchColor = [255, 0, 0],
  singlePointColor = [0, 255, 0],
  flags = 2
)

res_labels = ['sift', 'akaze', 'orb']
res_images = [sift_res, akaze_res, orb_res]

for i, (lbl, img) in enumerate(zip(res_labels, res_images)):
  plt.subplot(2, 2, i + 1)
  plt.imshow(img, cmap='gray')
  plt.title(lbl)

plt.show()