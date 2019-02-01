import cv2
from imutils.video import WebcamVideoStream
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--count", default=1, type=int, help="number of people to register")
args = vars(ap.parse_args())

def predict():
    face_classifier = cv2.CascadeClassifier("C:/Users/uvss/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("trainer/training_data.yml")
    vs = WebcamVideoStream("rtsp://root:essi@192.168.1.90/axis-media/media.amp").start()

    while True:
        image = vs.read()
        img=image.copy()
        gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale (gray, 1.3, 5)
        if faces is ():
            continue
        for(x, y, w, h) in faces:
            cv2.rectangle (img,(x, y),(x+w, y+h),(102, 255, 102),2)
            roi_gray = gray[y: y+h, x: x+w]
            label, conf = face_recognizer.predict(roi_gray)
            print(label, conf*100)
            cv2.putText(img, str(label), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow("frames", img)
        if cv2.waitKey(1) == ord("q"):
            break
    vs.stop()
    cv2.destroyAllWindows()
predict()