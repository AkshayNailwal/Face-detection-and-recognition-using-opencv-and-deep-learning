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
    faces = face_classifier.detectMultiScale (gray, 1.3, 5)
    if faces is ():
        return None
    for(x, y, w, h) in faces:
        cv2.rectangle (img,(x, y),(x+w, y+h),(102, 255, 102),2)
        roi_gray = gray[y: y+h, x: x+w]
        face_array.append(cv2.resize(gray[y:y+h, x:x+w],(100,100)))
    return face_array

def training_Data(path):
    sampled_faces=[]
    sampled_labels=[]
    img_name=None
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

    vs = WebcamVideoStream(src="rtsp://root:essi@192.168.1.90/axis-media/media.amp")
    if vs.grabbed:
        vs = vs.start()
        while(True):
            frame = vs.read()
            cv2.imshow("frame",frame)
            if cv2.waitKey(10) == ord("q"):
                break
            faces = face_Detector(frame)
            if faces is None or faces==[]:
                continue
            for face in faces:
                img_id+=1
                cv2.imshow("face",face)
                img_name=usr_dir + str(img_id) + ".jpg", face
                cv2.imwrite(img_name, face)
                sampled_labels.append(label)
                sampled_faces.append(face)
            if img_id>=50:
                break

    vs.stop()
    cv2.destroyAllWindows()
    return (sampled_faces, sampled_labels)

def confirm_save(recognizer, faces, labels, imageFolder):
    print("[Info] Saving Data.......")
    if os.path.exists("trainer/training_data.yml")==False:
        print("lll")
        recognizer.train(faces, np.array(labels))
        recognizer.save("trainer/training_data.yml")
        print("[Info] Data Saved")
    else:
        new_faces=[]
        new_labels=[]
        data_path = os.listdir(imageFolder)
        for usr_data in data_path:
            list_face_path = imageFolder+"/"+usr_data
            list_faces = os.listdir(list_face_path)
            for face in list_faces:
                new_faces.append(cv2.imread(list_face_path+"/"+face, 0))
                new_labels.append(usr_data[1:])
        print(len(new_faces),len(new_labels))
        print(np.concatenate(new_faces, axis = 0))
        recognizer.update(np.stack(new_faces, axis = 0), np.array(new_labels, dtype=np.int32))
        print(np.concatenate(new_faces, axis = 0).size)
        recognizer.save("trainer/training_data.yml")
        print("[Info] Data Saved")

def train(path, recognizer=None):
    face_recognizer = recognizer or cv2.face.LBPHFaceRecognizer_create()
    faces, labels = training_Data(path)
    print(len(faces), len(labels))
    if faces and labels:
        print(labels)
    confirm_save(face_recognizer, faces, labels, path)

time.sleep(1)     
train("dataset")