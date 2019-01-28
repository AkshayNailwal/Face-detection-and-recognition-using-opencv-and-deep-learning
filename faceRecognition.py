import cv2
face_classifier = cv2.CascadeClassifier("C:/Users/uvss/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

samples = 0
user_id=1
def face_detector (image, size=0.5):
    user_id+=1
    img=image.copy()
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale (gray, 1.25, 5)
    if faces is ():
        return img
    for(x, y, w, h) in faces:
        samples+=1
        cv2.rectangle (img,(x, y),(x+w, y+h),(102, 255, 102),2)
        roi_gray = gray[y: y+h, x: x+w]
        cv2.imwrite("dataset/"+"u"+str(user_id)+"."+str(samples)+".jpg",image[y:y+w, x:x+w])
    x=1
    return img

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = face_detector(frame)
    cv2.imshow('frame',gray)
    cv2.waitKey(1)
    if samples>10:
        break
cap.release()
cv2.destroyAllWindows()
