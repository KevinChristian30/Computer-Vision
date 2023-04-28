import cv2
import os
import numpy as np
import math

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Train
train_path = 'images/train'
tdir = os.listdir(train_path)

face_list = []
class_list = []

for idx, train_dir in enumerate(tdir):
  for image_path in os.listdir(f'{train_path}/{train_dir}'):
    path = f'{train_path}/{train_dir}/{image_path}'
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    faces = classifier.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)

    if len(faces) < 1:
      continue
    else:
      for face_rect in faces:
        x, y, w, h = face_rect
        face_image = gray[y:y+w, x:x+h]
        face_list.append(face_image)
        class_list.append(idx)
    
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_list, np.array(class_list))

# Test
test_path = 'images/test'

for path in os.listdir(test_path):
  fullpath = f'{test_path}/{path}'
  image = cv2.imread(fullpath)
  igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  faces = classifier.detectMultiScale(igray, scaleFactor = 1.2, minNeighbors = 5)

  if len(faces) < 1:
    continue
  
  for face_rect in faces:
    x, y, w, h = face_rect
    face_image = igray[y:y+w, x:x+h]
    res, conf = face_recognizer.predict(face_image)

    conf = math.floor(conf * 100) / 100
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    image_text = f'{tdir[res]} : {str(conf)}%'

    cv2.putText(image, image_text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 1)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
cv2.destroyAllWindows()