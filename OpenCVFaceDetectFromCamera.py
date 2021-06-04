# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:40:39 2021
OpenCV检测视频中的人脸
@author: YuWei
"""


import cv2
#import numpy as np


print("Build with OpenCV " + cv2.__version__)

cap = cv2.VideoCapture(0)

#cv2.namedWindow("PIC", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制

face_cascade = cv2.CascadeClassifier(r'./xml/haarcascade_frontalface_default.xml')

while(True):
    ret, frame = cap.read()                        #读取帧
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #灰度化展示
    faces = face_cascade.detectMultiScale(
       gray,
       scaleFactor = 1.15,
       minNeighbors = 5,
       minSize = (32,32)
    )
    print ("发现{0}个人脸!".format(len(faces)))

    for faceRect in faces:
        x, y, w, h = faceRect
        cv2.rectangle(frame,(x - 10, y - 10), (x + w + 10, y + h + 10), (0,255,0), 2)
       
    cv2.imshow("PIC", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):          #按‘q’退出
        break


cap.release()
cv2.destroyWindow("PIC")