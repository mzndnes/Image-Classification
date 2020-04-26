import cv2
import os
import time
import gc
import numpy as np

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
            img = cv2.imread(fl_nm, 0)
            #img = cv2.resize(img, (160, 160))
            face_img.append(img)
            lble.append(cl)

        cl += 1

    # return the array of faces and corresponding labels
    return np.array(face_img), np.array(lble)

def train_faces(face,lble):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face,np.array(lble))
    return face_recognizer


def main():
    start=time.time()

    dr='train'
    train,train_cl = readimg(dr)

    dr = 'test'
    test,test_cl = readimg(dr)

    model=train_faces(train,train_cl)

    tp=0
    for i in range(len(test)):
        # returns prediction and confidence
        plble,cf=model.predict(test[i])
        # if cf>37.0:
        #     tp+=1
        if plble==test_cl[i]:
            tp+=1

    print('Accuracy=',tp/len(test))



    end=time.time()
    print('Time taken=',end-start)

    gc.collect()


if __name__ == "__main__":
    main()



