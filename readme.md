### Overview:

In previous lesson, we learn about detecting the location of human face in images which is very important step of face recognition system. In this lesson, we will train Local Binary Patterns Histogram (LBPH) classifier to recognize the identity of face.

### Introduction to face recognizer:

We have already detected face in a image in previous lesson. Now, we will train face recognizer with these faces. OpenCV provides three different face recognizers and they are:

1. EigenFaces recognizer
2. FisherFaces recognizer
3. LBPH recognizer

Even though EigenFaces and FisherFaces are fast in recognition, their accuracy suffers from contrast in the image. We will not talk about these methods in detail. The accuracy of LBPH does not get affected with the contrast in the image. So, we will use LBPH recognizer for our purpose.

### Training LBPH face recognizer:

Unlike EigenFaces that extract principal features from all images, LBPH extracts local features of an image by comparing each pixel with its neighboring pixels. LBPH considers the window of 3 x 3 pixels as shown in the figure below. 

![LBPH](/home/dnes/PycharmProjects/tenflw2/1302.png)

To extract the local feature of center pixel, it changes the neighboring pixels into 1, if it is greater than center pixel value otherwise 0. The local feature of center pixel is the decimal value of the clockwise order of the neighboring pixels values as shown in figure above. This is done on entire images. The face constructed with local features does not get affected with light. Histogram is constructed from these features. We define train_faces function to train the LBPH recognizer as follows

```python
def train_faces(face,lble):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face,np.array(lble))
    return face_recognizer
```

First we should create face_recognizer object of LBPH and then train it with faces and corresponding labels. The labels must be strictly in numpy array.

### Recognizing the test faces:

Now we are ready to recognize the test faces. We predict the class of test face using the predict function of LBPH object as shown below.

```python
plble,cf=model.predict(test[0])
```

This function returns the label of the face and its prediction confidence. We evaluate the accuracy of LBPH as the fraction of correctly predicated test faces as shown below.

```python
    tp=0
    for i in range(len(test)):
        plble,cf=model.predict(test[i])
        if plble==test_cl[i]:
            tp+=1
    print('Accuracy=',tp/len(test))
```

We used dataset of 198 images of six bollywood actors that was created in previous lesson. The training set of 22 images of each actor were used as training set and 11 were used as test set. The LBPH face recognizer was able to secure the accuracy of 65% on this dataset of actors.
