import numpy as np
import pickle
import os
import cv2
import time
import datetime
import imutils

curr_path = os.getcwd()

print("Loading face detection model")
proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

recognizer = pickle.loads(open('recognizer.pickle', "rb").read())
le = pickle.loads(open('le.pickle', "rb").read())
print("Starting test video file")
vs = cv2.VideoCapture("testvideo.mp4")
#vs=cv2.VideoCapture(0)
time.sleep(1)

while True:

    ret, frame = vs.read()
    frame = imutils.resize(frame, width=1500)

    (h, w) = frame.shape[:2]

    image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    face_detector.setInput(image_blob)
    face_detections = face_detector.forward()

    for i in range(0, face_detections.shape[2]):

        '''here we taken only the face which one have accuracy 50% and convert the imageblob which is stored in the face_detections
               after the encoding process we are going to blob the image by resizeing it into 96,96 image and store the image in the 
               face_recognizer to recognize the face'''

        confidence = face_detections[0, 0, i, 2]

        if confidence >= 0.5:
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            (fH, fW) = face.shape[:2]

            face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), True, False)

            face_recognizer.setInput(face_blob)
            vec = face_recognizer.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            #text = "{}: {:.2f}".format(name, proba * 100)
            text = "{}".format(name)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            if(text=="adam 18_07_1980"):
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print("Class A adam 18_07_1980")
            elif(text=="mary21_03_2000"):
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print("Class B mary21_03_2000")
            elif(text=="mathew20_6_1990"):
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print("Class B mathew20_6_1990")
            else:
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print("Class C Other Customer")



    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()



