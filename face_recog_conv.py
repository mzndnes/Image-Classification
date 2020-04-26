import numpy as np
import cv2
import os
import tensorflow as tf


# read images and extract face
def readimg(img_dir):
    # get current directory
    curpath = os.path.abspath(os.getcwd())
    # get directory of images to be read
    test_train_path = os.path.join(curpath, 'images', 'face94_' + img_dir)
    # get list of sub directories
    dirs = os.listdir(test_train_path)

    face_img = []
    lble = []
    cl = 0

    for dr in dirs:

        fl_dr = os.path.join(test_train_path, dr)
        files = os.listdir(fl_dr)

        for fl in files:

            fl_nm = os.path.join(fl_dr, fl)
            img = cv2.imread(fl_nm, 0)

            f_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            face = f_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)
            
            if len(face) != 1:
                continue
            (x, y, w, h) = face[0]
            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (160, 160))
            face=np.reshape(face,(160,160,1))
            # print(face)
            # input()

            # cv2.imshow('p2',face)
            # cv2.waitKey(6000)
            # cv2.destroyAllWindows()

            face_img.append(face)
            lble.append(cl)

        cl += 1

    # return the array of faces and corresponding labels
    return face_img, lble

def face_recognizer(s,c,nc):
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(s,s,c)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(nc, activation='softmax')
    ])
    return model

def main():
    #size, channel of images
    s=160
    c=1
    nc=10
    dr = 'train'
    train, train_cl = readimg(dr)
    train=np.array(train)
    train =train/ 255.0
    dr = 'test'
    test, test_cl = readimg(dr)
    test = np.array(test)
    test=test/255.0
    model=face_recognizer(s,c,nc)
    model.summary()

    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(np.array(train),np.array(train_cl), epochs=26)
    test_loss, test_accuracy = model.evaluate(np.array(test), np.array(test_cl))
    print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy * 100))

    # print(train[0].shape)
    # print(test[0].shape)

if __name__ == "__main__":
    main()
