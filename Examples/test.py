import cv2
cv2.imshow("res",cv2.imread("challange.jpg"))
if cv2.waitKey(0) == "0xFF":
    cv2.destroyALlWindows()



img=cv2.imread("C:/Users/uvss/Desktop/test2.jpg")
res=face_detector(img)
cv2.resize(res,(img.shape[0],img.shape[1]))
cv2.imshow("res",res)
cv2.waitKey (0)
cv2.destroyAllWindows ()
