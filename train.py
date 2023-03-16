import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FaceLoading:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(img)[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, dir):
        faces = []
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                single_face = self.extract_face(path)
                faces.append(single_face)
            except Exception as e:
                pass
        return faces

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            faces = self.load_faces(path)
            labels = [sub_dir for _ in range(len(faces))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(faces)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)

# Load faces and their corresponding labels
face_loading = FaceLoading("data")
X, Y = face_loading.load_classes()

# Compute face embeddings using FaceNet
embedder = FaceNet()
embedded_X = []
for img in X:
    embedded_X.append(embedder.embeddings(np.expand_dims(img.astype('float32'), axis=0))[0])
embedded_X = np.asarray(embedded_X)

# Save the face embeddings and their corresponding labels to a compressed numpy file
np.savez_compressed('faces_embeddings.npz', embedded_X, Y)

# Encode labels using LabelEncoder and split the data into train and test sets
encoder = LabelEncoder()
encoder.fit(Y)
Y_encoded = encoder.transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(embedded_X, Y_encoded, shuffle=True, random_state=17)

# Train a linear SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

# Predict labels for train and test sets and compute their accuracy
ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)
train_accuracy = accuracy_score(Y_train, ypreds_train)
test_accuracy = accuracy_score(Y_test, ypreds_test)

# Save the trained model to a pickle file
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)