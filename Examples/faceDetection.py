import cv2
import time
from imutils.video import WebcamVideoStream
import imutils

face_classifier = cv2.CascadeClassifier("C:/Users/uvss/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
def face_detector (image, id):
    img=image.copy()
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale (gray, 1.3, 5)
    if faces is ():
        return img
    for(x, y, w, h) in faces:
        cv2.rectangle (img,(x, y),(x+w, y+h),(102, 255, 102),2)
        roi_gray = gray[y: y+h, x: x+w]
        cv2.imwrite("detected/Haar/img"+str(id)+".jpg",cv2.resize(image[y:y+h, x:x+w],(100,100)))
    x=1
    return img

def detect_faces():
    print("[INFO] starting video stream...")
    vs = WebcamVideoStream(src="rtsp://root:essi@192.168.1.90/axis-media/media.amp")
    if vs.grabbed:
        vs=vs.start()
        time.sleep(2.0)
        print("------started survillancing------")
        id = 0
        x=1
        while(True):
            x+=1
            frame = vs.read()
            id+=1
            if x%2==0:
                gray = face_detector(frame, id) 
            cv2.imshow('frame',gray)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        vs.stop()
        cv2.destroyAllWindows()

detect_faces()