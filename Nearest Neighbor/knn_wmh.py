import numpy as np
import cv2
import gc
import os
import csv
import wmh

def readimg(img_dir):

    curpath = os.path.abspath(os.getcwd())

    #rfl text file contains the names of image files and its label
    rfl = os.path.join(curpath, img_dir + '.txt')

    #open text file
    csvf = open(rfl, 'r')
    csvRD = csv.reader(csvf, delimiter=',')

    data=[]
    for row in csvRD:
        rdata = []
        flg = 0
        for col in row:
            if flg == 0:
                flg = 1
                fnm = str(col) + '.jpg'
            else:
                #label of image file
                cl = int(col)
                #push the label of image file into rdata list
                rdata.append(cl)
                #image file
                img_fl = os.path.join(curpath, 'img_' + img_dir, fnm)
                # read image in grayscale
                img = cv2.imread(img_fl, 0)
                h=img.shape
                # convert two dimentional image into one dimentional list
                rc= np.resize(img, h[0]*h[1])

                for i in range(len(rc)):
                    rdata.append(rc[i])
        #push image to data list
        data.append(rdata)

    #close text file
    csvf.close()
    return data


def main():

    k = 1
    h=10



    dr='train'
    train = readimg(dr)

    dr = 'test'
    test = readimg(dr)

    print('hi')
    trinst=len(train)
    tinst=len(test)
    dim=len(train[0])-1

    hasher=wmh.wmh_hash(train,test,h,k)

    trhash = hasher.gen_hash(train)
    thash = hasher.gen_hash(test)

    hasher.memorize(trhash)
    print(hasher.predict(thash[0]))

    gc.collect()


if __name__ == "__main__":
    main()
