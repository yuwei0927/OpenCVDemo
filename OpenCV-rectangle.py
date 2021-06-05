# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:53:39 2021
rectangle方法使用说明
@author: YuWei
"""

import cv2

print("Build with OpenCV " + cv2.__version__)

#打开图片
img = cv2.imread(r'./res/pic/1.jpg', cv2.IMREAD_REDUCED_COLOR_4)
print(img.shape)

#图像矩阵大小
print(img.size)

#图像的数据类型
print(img.dtype)


cv2.namedWindow("PIC", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制

'''
cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) 
cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
cv2.rectangle的pt1 和 pt2 参数分别代表矩形的左上角和右下角两个点，相对于(0,0)，而且 x 坐标轴是水平方向的，y 坐标轴是垂直方向的。
color 对应(B, G, R)
thickness 参数表示矩形边框的厚度，如果为负值，如 CV_FILLED(-1)，则表示填充整个矩形

------------------------------------------------>x
|
|
|    x1,y1---------------------
|      |                      |
|      |                      |
|      |                      |
|      |                      |
|      |                      |
|      |                      |
|      |--------------------x2,y2
|
|
|
∨
y
'''
color = (0,255,255)
p1 = (100, 100)
p2 = (200, 200)
thickness = 2

cv2.rectangle(img, p1, p2, color, thickness)

cv2.imshow("PIC", img)

cv2.waitKey(0)

cv2.destroyWindow("PIC")
