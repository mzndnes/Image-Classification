import cv2
img=cv2.imread('bean.jpg')
f_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = f_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('Faces Detected',img)
cv2.waitKey(6000)
cv2.destroyWindows()