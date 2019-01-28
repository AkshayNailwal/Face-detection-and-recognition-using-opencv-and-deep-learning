import cv2
import numpy as np
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier("C:/Users/uvss/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
def face_detector (img, size=0.5):
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale (gray, 1.5, 5)
    if faces is ():
        return img

    for(x, y, w, h) in faces:
        cv2.rectangle (img,(x+10, y),(x+w-10, y+h),(255, 0, 0),2)
        roi_gray = gray[y: y+h, x: x+w]

    return img

cap = cv2.VideoCapture("test2.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = face_detector(frame)
    
    cv2.imshow('frame',gray)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
