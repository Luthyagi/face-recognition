# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:20:05 2019
@author: manishluthyagi
"""

import cv2
import pickle
import numpy as np
from keras.preprocessing import image
import timeit

f = open('classifier1.pickle','rb')
classifier = pickle.loads(f.read())
f.close()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

while True:
    
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)
        
        test_image = cv2.resize(frame,(64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis = 0)
        result = classifier.predict(test_image)
        if result[0][0] >= 0.5:
            text = 'udit'
        else :
            text = 'manish'
            
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()










