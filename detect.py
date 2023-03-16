import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# Initialize FaceNet
facenet = FaceNet()

# Load precomputed face embeddings
faces_embeddings = np.load("faces_embeddings.npz")
X = faces_embeddings['arr_0']
Y = faces_embeddings['arr_1']

# Encode labels using sklearn's LabelEncoder
encoder = LabelEncoder()
encoder.fit(Y)

# Load Haar Cascade classifier for face detection
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load SVM model for face recognition
model = pickle.load(open("svm_model.pkl", 'rb'))

# Initialize video capture
cap = cv.VideoCapture(1)

# Run face recognition in a loop
while cap.isOpened():
    # Read a frame from the video capture
    _, frame = cap.read()

    # Convert frame to RGB and grayscale
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    # Iterate through detected faces
    for x, y, w, h in faces:
        # Crop and resize face image to 160x160 for FaceNet
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160, 160))

        # Compute face embedding using FaceNet
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)

        # Recognize face using SVM model
        face_name = model.predict(ypred)

        # Convert label encoding back to original names
        final_name = encoder.inverse_transform(face_name)[0]

        # Draw rectangle and label on face in original frame
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
        cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 3, cv.LINE_AA)

    # Show the frame with face recognition annotations
    cv.imshow("Face Recognition:", frame)

    # Exit if 'q' key is pressed
    if cv.waitKey(1) & ord('q') == 27:
        break

# Release video capture and close windows
cap.release()
cv.destroyAllWindows()
