import os
import cv2
import numpy as np
import time
import imutils
from imutils.video import WebcamVideoStream

face_classifier = cv2.CascadeClassifier("C:/Users/uvss/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
def face_Detector (image):
    face_array = []
    img=image.copy()
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale (gray, 1.5, 5)
    if faces is ():
        return None
    for(x, y, w, h) in faces:
        cv2.rectangle (img,(x, y),(x+w, y+h),(102, 255, 102),2)
        roi_gray = gray[y: y+h, x: x+w]
        face_array.append(cv2.resize(gray[y:y+h, x:x+w],(100,100)))
        print(len(face_array))
    return face_array

def training_Data(path):
    sampled_faces=[]
    sampled_labels=[]
    usr_id, label = 1, 1
    img_id = 0
    users=os.listdir(path)
    users.sort(key = lambda x: int(x[1:])) # sorting all the user data directories
    if users==[]:
        usr_dir = path + "/s" + str(usr_id) + "/"
        os.mkdir(usr_dir)
    else:
        label = int(users[-1][1:])+1
        usr_id = str(label)  # get [1:] part of last element in the list
        usr_dir = path + "/s" + usr_id + "/"
        os.mkdir(usr_dir)

    cap = WebcamVideoStream(src="rtsp://root:essi@192.168.1.90/axis-media/media.amp").start()

    while(True):
        frame = cap.read()
        cv2.imshow("frame",frame)
        if cv2.waitKey(10) == ord("q"):
            break
        faces = face_Detector(frame)
        if faces is None or faces==[]:
            continue
        for face in faces:
            img_id+=1
            cv2.imshow("face",face)
            cv2.imwrite(usr_dir + str(img_id) + ".jpg", face)
            sampled_labels.append(label)
            sampled_faces.append(face)
        if img_id>=10:
            break

    cap.stop()
    cv2.destroyAllWindows()
    return (sampled_faces, sampled_labels)

def train(path, recognizer=None):
    face_recognizer = recognizer or cv2.face.LBPHFaceRecognizer_create()
    faces, labels = training_Data(path)
    print(len(faces), len(labels))
    if faces and labels:
        face_recognizer.train(faces, np.array(labels))
        face_recognizer.save("trainer/training_data.yml")

train("dataset")