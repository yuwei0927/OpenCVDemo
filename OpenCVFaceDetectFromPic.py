# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:54:11 2021

@author: YuWei
"""


import cv2
#import numpy as np


print("Build with OpenCV " + cv2.__version__)

img = cv2.imread(r'./pic/face.jpg', cv2.IMREAD_REDUCED_COLOR_4 )#原图1/4显示

#cv2.namedWindow("PIC", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(r'./xml/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(
   gray,
   scaleFactor = 1.15,
   minNeighbors = 5,
   minSize = (32,32)
)


print ("发现{0}个人脸!".format(len(faces)))

for faceRect in faces:
    x, y, w, h = faceRect
    cv2.rectangle(img,(x - 10, y - 10), (x + w + 10, y + h + 10), (0,255,0), 1)
   
cv2.imshow("PIC", img)

cv2.waitKey(0)

cv2.destroyWindow("PIC")