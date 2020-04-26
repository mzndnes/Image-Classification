import numpy as np
import cv2
import os

def img_crop(img_dir):
    # get current directory
    curpath = os.path.abspath(os.getcwd())
    # get directory of images to be read
    src_dr = os.path.join(curpath, 'images',  img_dir)

    #cropped directory
    dst_dr = os.path.join(curpath, 'images', 'cropped')


    try:
        os.mkdir(dst_dr)
    except:
        print('Exists')

    dirs = os.listdir(src_dr)

    for dr in dirs:

        src_fl_dr = os.path.join(src_dr, dr)
        dst_fl_dr=os.path.join(dst_dr,dr)
        files = os.listdir(src_fl_dr)

        try:
            os.mkdir(dst_fl_dr)
        except:
            print('Exists')

        for fl in files:

            src_fl = os.path.join(src_fl_dr, fl)
            dst_fl = os.path.join(dst_fl_dr, fl)
            gray_img = cv2.imread(src_fl, 0)
            color_img=cv2.imread(src_fl, 1)

            f_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            face = f_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

            if len(face) != 1:
                continue
            (x, y, w, h) = face[0]
            face = color_img[y:y + h, x:x + w,:]
            face = cv2.resize(face, (160, 160))
            cv2.imwrite(dst_fl,face)

            # cv2.imshow('old', color_img)
            # cv2.imshow('new',face)
            # cv2.waitKey(6000)
            # cv2.destroyAllWindows()

def main():
    crop_img('act')

if __name__ == "__main__":
    main()
