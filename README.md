# Face-recognition
The face recognition program uses Facenet, MTCNN, and SVM for training and Haarcascade, Facenet, and SVM model for face recognition

During the training process, MTCNN detects and aligns facial features, Facenet generates feature vectors, and SVM is trained to match the vectors with known identities. The resulting model is then used in the face recognition code, which employs Haarcascade for face detection, Facenet for feature generation, and the SVM model for identity matching. This system provides accurate and reliable face recognition capabilities, and can be used in a variety of applications.

## Training

**1. Data preparation**

In order to train a face recognition system, you have to collect a certain number of pictures that capture the faces of the people you want the system to recognize. You need to have a minimum of two distinct image classes. The dataset folder structure in Facenet may look like this:

```
dataset/
    person1/
        img1.jpg
        img2.jpg
        ...
    person2/
        img1.jpg
        img2.jpg
        ...
    ...
```

**2. Train**

To get started, you'll have to locate line 55 in the train.py file and change the value of the ```'dataset'``` field to the path of your dataset directory.
