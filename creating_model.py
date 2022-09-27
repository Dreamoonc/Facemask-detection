import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

img= cv2.imread('image.jpeg')
print(img.shape)
print(img[0])
plt.imshow(img)
# while True:
#     cv2.imshow('result',img)
#     if cv2.waitKey(2)==27:
#         break
# cv2.destroyAllWindows()
data=cv2.CascadeClassifier('data.xml')
data.detectMultiScale(img)

while True :
    faces=data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
    cv2.imshow('result',img)
    if cv2.waitKey(2) ==27:
        break
cv2.destroyAllWindows()


capture=cv2.VideoCapture(0)
faceData=[]
while True:
    flag,img=capture.read()
    if flag:
        faces=data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(50,50))
            print(len(faceData))
            if len(faceData)<1000:
                faceData.append(face)
        cv2.imshow('result',img)
        if cv2.waitKey(2)==27 or len(faceData)>= 1000:
            break
capture.release()
cv2.destroyAllWindows()

np.save('without_mask.npy',faceData)

time.sleep(30)

capture=cv2.VideoCapture(0)
faceData=[]
while True:
    flag,img=capture.read()
    if flag:
        faces=data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(50,50))
            print(len(faceData))
            if len(faceData)<1000:
                faceData.append(face)
        cv2.imshow('result',img)
        if cv2.waitKey(2)==27 or len(faceData)>= 1000:
            break
capture.release()
cv2.destroyAllWindows()

np.save('with_mask.npy',faceData)