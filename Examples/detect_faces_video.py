#  -----------------------------USAGE------------------------------
# python detect_faces_video.py --prototxt models/deploy.prototxt.txt --model models/res10_300x300_ssd_iter_140000.caffemodel

import numpy as np
import argparse
import time
import cv2
from imutils.video import WebcamVideoStream
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = WebcamVideoStream(src="rtsp://root:essi@192.168.1.90/axis-media/media.amp").start()
time.sleep(2.0)
print("[INFO] ------started survillancing------")
id = 0
while True:
    img = vs.read()
    frame = img.copy()
    (h, w, l) = frame.shape
    id = id+1
    if frame.any():
        pass
    else:
        print("frame does not exists")
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # current time
    c = int(time.time())
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        if confidence < args["confidence"]:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX+5, endY-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        try:
            if frame[startY:endY, startX:endX].size:
                cv2.imwrite("detected/Dnn/img"+str(id)+".jpg", cv2.resize(img[startY:endY, startX:endX], (100, 100)))
        except Exception as err:
            print(err.args)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        id=0
        break

cv2.destroyAllWindows()
vs.stop()