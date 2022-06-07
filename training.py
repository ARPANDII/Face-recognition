import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os
import imutils

curr_path = os.getcwd()
'''system call which is used to get the current directory'''

print("Loading face detection model")

'''now we are going to import the buildin file which is prototype,caffemodel 
the model what we are using is to detect the pic is caffe'''

proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")

'''then here we'll import the recognition the face using buildin file and we are going to 
 recognize the face using torch'''

recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

'''now we are going to open the pickle file what we saved already during the execution of the training.py model'''

data_base_path = os.path.join(curr_path, 'database_new')

filenames = []
for path, subdirs, files in os.walk(data_base_path):
    for name in files:
        filenames.append(os.path.join(path, name))

face_embeddings = []
face_names = []

for (i, filename) in enumerate(filenames):
    print("Processing image {}".format(filename))

    image = cv2.imread(filename)
    image = imutils.resize(image, width=600)
    ''' this one is to resize the frame for better quality'''

    (h, w) = image.shape[:2]

    image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    '''after resizeing the images we have to blob the image for recognize'''

    face_detector.setInput(image_blob)
    face_detections = face_detector.forward()

    i = np.argmax(face_detections[0, 0, :, 2])
    confidence = face_detections[0, 0, i, 2]

    if confidence >= 0.5:

        box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = image[startY:endY, startX:endX]

        face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0), True, False)

        face_recognizer.setInput(face_blob)
        face_recognitions = face_recognizer.forward()

        name = filename.split(os.path.sep)[-2]

        face_embeddings.append(face_recognitions.flatten())
        face_names.append(name)


data = {"embeddings": face_embeddings, "names": face_names}

le = LabelEncoder()
labels = le.fit_transform((data["names"]))

recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open('recognizer.pickle', "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open("le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
