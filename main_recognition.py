import os
import sys
import cv2
import numpy as np

faceCascade=cv2.CascadeClassifier('E:\Programs\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
ret,frame = video_capture.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
  gray,
  scaleFactor=1.2,
  minNeighbors=4,
  minSize=(30,30)
)
size = len(faces)
face_resized = []
for (x, y, w, h) in faces:
    #transform face - resize + gray color
    sideSquare = max([width,height])
    face = cv2.getRectSubPix(gray,(sideSquare,sideSquare),  (x +(w /2), (y +h /2)))
    face_resized.append(cv2.resize(face,(width,height),1.0,1.0,0))
    cv2.rectangle(frame, (x, y), (x +w, y +h), (0, 255, 0), 2)
    cv2.imshow('Webnet face detection', frame)
model = cv2.createLBPHFaceRecognizer(threshold=200)
model.load('E:\Programs\opencv\sources\data\lbpcascadeslbpcascade_frontalcatface.xml')

   face = cv2.imread('C:\Users\Admin\Documents')
   unlink('C:\Users\Admin\Documents')
   face_resized = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
   face_resized = cv2.resize(face,(width,height),1.0,1.0,0)
   [p_label, p_confidence] = model.predict(face_resized)
   if p_label == -1:
       p_confidence = 0
       name = -1
       print name
   else:
      name = p_label
      print "Predicted label = %s (confidence=%.2f)" % ( name,p_confidence)
      logFaceRecog(name,p_confidence)
