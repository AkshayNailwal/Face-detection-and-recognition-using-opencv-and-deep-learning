import cv2
face_classifier = cv2.CascadeClassifier("C:/Users/uvss/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
def face_detector (image, id, size=0.5):
    img=image.copy()
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale (gray, 1.27, 5)
    if faces is ():
        return img
    for(x, y, w, h) in faces:
        cv2.rectangle (img,(x, y),(x+w, y+h),(102, 255, 102),2)
        roi_gray = gray[y: y+h, x: x+w]
        cv2.imwrite("detected/Haar/img"+str(id)+".jpg",cv2.resize(image[y:y+w, x:x+w],(100,100)))
    x=1
    return img

cap = cv2.VideoCapture("test2.mp4")
id = 0
x=1
while(cap.isOpened()):
    x+=1
    ret, frame = cap.read()
    id+=1
    if x%2==0:
        gray = face_detector(frame, id)
    cv2.imshow('frame',gray)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
