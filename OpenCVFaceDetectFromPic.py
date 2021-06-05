# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:54:11 2021
OpenCV检测图片中的人脸
@author: YuWei
"""


import cv2
#import numpy as np


print("Build with OpenCV " + cv2.__version__)

img = cv2.imread(r'./res/pic/5.jpg', cv2.IMREAD_REDUCED_COLOR_2 )#原图1/2显示

#cv2.namedWindow("PIC", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#加载对应的人脸分类器
face_cascade = cv2.CascadeClassifier(r'./xml/haarcascade_frontalface_alt2.xml')

#使用灰度图片来检测
'''
detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) -> objects
image--待检测图片，一般为灰度图像加快检测速度；
scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。
        默认为1.1即每次搜索窗口依次扩大10%,这个参数设置的越大，计算速度越快，但可能会错过了某个大小的人脸
minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。
        如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
        如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
        这种设定值一般用在用户自定义对检测结果的组合程序上；
'''
faces = face_cascade.detectMultiScale(
   gray,
   scaleFactor = 1.1,  
   minNeighbors = 5,
   minSize = (30,30)
)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

print ("发现{0}个人脸!".format(len(faces)))

for num in range(len(faces)):
    x, y, w, h = faces[num]
    cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), (0,255,0), 1)
    cv2.putText(img, "{0}".format(num), (x - 10, y - 10), font, 1, (0, 0, 255), 1)

   
cv2.imshow("PIC", img)

cv2.waitKey(0)

cv2.destroyWindow("PIC")