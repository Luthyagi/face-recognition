# -*- coding: utf-8 -*-
"""
program for creating a dataset for program

Created on Thu Apr 18 22:20:05 2019

@author: manishluthyagi
"""

import cv2
import os

classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
image_count = 0
person_name = input("\n Person name in Video stream :\t >")

os.mkdir("dataset/"+person_name)
while True:
    
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    

    # Drawing a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)
        
    # Display the resulting frame
    cv2.imwrite("dataset/"+person_name+'/'+str(image_count)+ ".jpg", frame[ y:y+h, x:x+w])
    cv2.imshow(person_name,frame)
    image_count = image_count + 1
    if image_count > 200:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()