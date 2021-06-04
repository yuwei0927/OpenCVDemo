# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:53:39 2021
OpenCV打开图片，显示图片信息，并分离出RGB三个通道显示出来
@author: YuWei
"""

import cv2

print("Build with OpenCV " + cv2.__version__)

#打开图片
#img = cv2.imread("ecarx.png", cv2.IMREAD_REDUCED_GRAYSCALE_2 )#灰度显示，并调整到原图的1/2
#img = cv2.imread("ecarx.png", cv2.IMREAD_GRAYSCALE)#灰度显示，原图大小不变
#img = cv2.imread("ecarx.png", cv2.IMREAD_COLOR)#RGB彩色显示，原图大小不变
#img = cv2.imread("ecarx.png", cv2.IMREAD_UNCHANGED)#原图显示，原图大小不变
img = cv2.imread("FadeInOut.jpg", cv2.IMREAD_UNCHANGED)#原图显示，原图
'''
PNG格式和JPG格式编码方式不同
'''

'''
图像矩阵的shape属性表示图像的大小，shape会返回tuple元组
第一个元素表示矩阵行数(高度)，第二个元组表示矩阵列数(行数)，第三个元素表示通道。
'''
print(img.shape)

#图像矩阵大小
print(img.size)

#图像的数据类型
print(img.dtype)

#进行通道分离（四通道）。为什么是四个通道？因为PNG格式的图片格式为BGRA
if img.shape[-1] == 4:
    b,g,r,a = cv2.split(img)
else:
    b,g,r = cv2.split(img)



cv2.namedWindow("PIC", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
#cv2.namedWindow("PIC", cv2.WINDOW_NORMAL )#用户可以调整窗口大小（无限制）/也可以用于将全屏窗口切换到正常大小
cv2.namedWindow("Blue", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
cv2.namedWindow("Green", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
cv2.namedWindow("Red", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
if img.shape[-1] == 4:
    cv2.namedWindow("a", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
cv2.imshow("PIC", img)

#cv2.imshow("Blueb", b)
#cv2.imshow("Green", g)
#cv2.imshow("Red", r)
#if img.shape[-1] == 4:
#    cv2.imshow("a", a)
#虽然分离出来了4个通道，但是每个通道只是只是针对当前色彩的一个颜色值，单独展示的时候，不包含色彩信息，因此全是灰度，也就是黑白色
#cv2.waitKey(0)


imgBlue = img.copy()
imgGreen = img.copy()
imgRed = img.copy()

#显示蓝色图层
imgBlue[:,:,1] = 0
imgBlue[:,:,2] = 0

#显示绿色图层
imgGreen[:,:,0] = 0
imgGreen[:,:,2] = 0

#显示红色图层
imgRed[:,:,0] = 0
imgRed[:,:,1] = 0

cv2.imshow("Blue", imgBlue)
cv2.imshow("Green", imgGreen)
cv2.imshow("Red", imgRed)

cv2.waitKey(0)

cv2.destroyWindow("PIC")
cv2.destroyWindow("Blue")
cv2.destroyWindow("Green")
cv2.destroyWindow("Red")
if img.shape[-1] == 4:
    cv2.destroyWindow("a")