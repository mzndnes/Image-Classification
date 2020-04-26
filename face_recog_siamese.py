import cv2
import numpy as np
import os
import tensorflow as tf
from embedding import emb

# def embedding(img):
#     model=load_model('facenet_keras.h5',compile=False)
#     return model.predict(img)[0]

def readimg(tr_t):
    # get current directory
    curpath = os.path.abspath(os.getcwd())
    # get directory of images to be read
    img_dir = os.path.join(curpath, 'images', 'act_' + tr_t)
    # get list of sub directories
    dirs = os.listdir(img_dir)

    face_img = []
    lble = []
    cl = 0

    for dr in dirs:

        fl_dr = os.path.join(img_dir, dr)
        files = os.listdir(fl_dr)

        for fl in files:
            fl_nm = os.path.join(fl_dr, fl)
            img = cv2.imread(fl_nm, 1)
            img = cv2.resize(img, (160, 160))
            face_img.append(img)
            lble.append(cl)

        cl += 1

    # return the array of faces and corresponding labels
    return np.array(face_img), np.array(lble)

def face_recognizer(nc):
    model = tf.keras.models.Sequential([

        tf.keras.layers.Dense(64, activation='relu', input_dim=128),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(nc, activation='softmax'),

    ])
    return model

def main():
    #incorrect value of nc may arise which is outside the valid range of [0, 5)
    nc=6
    dr = 'train'
    train, train_cl = readimg(dr)
    x_train = []

    y_train =train_cl
    # train =train/ 255.0

    e = emb()
    for x in train:
        img = x.astype('float') / 255.0
        img = np.expand_dims(img, axis=0)
        embs = e.calculate(img)
        x_train.append(embs)

    x_train = np.array(x_train, dtype='float')


    dr = 'test'
    test, test_cl = readimg(dr)

    x_test=[]
    y_test = test_cl

    for x in test:
        img = x.astype('float') / 255.0
        img = np.expand_dims(img, axis=0)
        embs = e.calculate(img)
        x_test.append(embs)

    x_test = np.array(x_test, dtype='float')


    model=face_recognizer(nc)
    model.summary()

    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train,y_train, epochs=26)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy * 100))

    # print(train[0].shape)
    # print(test[0].shape)

if __name__ == "__main__":
    main()
