# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:53:39 2021
图像缩放、平移、旋转、仿射、透视变换
https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
@author: YuWei
"""

import cv2
import numpy as np
import matplotlib.pylab  as plt

print("Build with OpenCV " + cv2.__version__)

'''
图片旋转
'''
def picXuanZhuan():
    img = cv2.imread(r'./res/pic/1.jpg', cv2.IMREAD_REDUCED_COLOR_4)
    print(img.shape)
    
    rows,cols,cnt = img.shape
    #getRotationMatrix2D有三个参数，第一个为旋转中心，第二个为旋转角度(逆时针方向)，第三个为缩放比例
    M = cv2.getRotationMatrix2D((cols/2,rows/2), 180, 1)
    
    dst = cv2.warpAffine(img, M, (cols,rows))
    
    print(dst.shape)
    
    cv2.namedWindow("original", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
    cv2.imshow("original", img)
    cv2.imshow("result", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#picXuanZhuan()

'''
图片缩放
对缩小，优选的interpolation方法：cv2.INTER_AREA该方法可以避免波纹的出现
对放大，优选的interpolation方法：cv2.INTER_CUBIC和cv2.INTER_LINEAR(默认)
'''
def picResize():
    img = cv2.imread(r'./res/pic/1.jpg', cv2.IMREAD_REDUCED_COLOR_4)
    print(img.shape)
    
    res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)#缩小到原来的0.5倍
    #res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)#放大到原来的2倍
    
    print(res.shape)
    
    cv2.namedWindow("original", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
    cv2.imshow("original", img)
    cv2.imshow("result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#picResize()

'''
图片平移
按（100,50）平移，也就是原来（0,0）平移到（100,50）
'''
def picMove():
    img = cv2.imread(r'./res/pic/1.jpg', cv2.IMREAD_REDUCED_COLOR_4)
    print(img.shape)
    
    rows,cols,cnt = img.shape
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))

    
    print(dst.shape)
    
    cv2.namedWindow("original", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
    cv2.imshow("original", img)
    cv2.imshow("result", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#picMove()

'''
图片仿射
仿射变换更直观的叫法可以叫做“平面变换”或者“二维坐标变换”.
仿射变换的方程组有6个未知数，所以要求解就需要找到3组映射点，三个点刚好确定一个平面
在仿射变换中，原始图像中的所有平行线仍将在输出图像中平行。
为了找到变换矩阵，我们需要输入图像中的三个点及其在输出图像中的相应位置。 
然后cv.getAffineTransform将创建一个2x3矩阵，该矩阵将传递给cv.warpAffine
'''
def picFangshe():
    img = cv2.imread(r'./res/pic/1.jpg', cv2.IMREAD_REDUCED_COLOR_4)
    print(img.shape)
    
    rows,cols,cnt = img.shape
    
    pts1 = np.float32([[0,0],[200,50],[50,200]])#原始图像中的数据点坐标
    pts2 = np.float32([[10,100],[200,50],[100,250]])#仿射后的图像对应的新坐标

    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()
    
    print(dst.shape)
    
    cv2.namedWindow("original", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
    cv2.imshow("original", img)
    cv2.imshow("result", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#picFangshe()

'''
图片透视
透视变换更直观的叫法可以叫做“空间变换”或者“三维坐标变换”.
透视变换的方程组有8个未知数，所以要求解就需要找到4组映射点，四个点就刚好确定了一个三维空间
透视变换，需要一个3x3变换矩阵。
即使在转换之后，直线仍将保持笔直. 要找到此变换矩阵，输入图像上需要4个点，输出图像上需要相应的点. 
在这4个点中，其中3个不应该共线. 
然后可以通过函数cv2.getPerspectiveTransform找到变换矩阵. 然后将cv2.warpPerspective应用于此3x3变换矩阵
'''
def picTouShi():
    img = cv2.imread(r'./res/pic/1.jpg', cv2.IMREAD_REDUCED_COLOR_4)
    print(img.shape)
    
    rows,cols,cnt = img.shape
    
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    
    dst = cv2.warpPerspective(img,M,(cols,rows))
    
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()
    
    print(dst.shape)
    
    cv2.namedWindow("original", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)#用户无法调整窗口大小，大小受显示的图像限制
    cv2.imshow("original", img)
    cv2.imshow("result", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

picTouShi()

# img = cv.imread('sudoku.png')
# rows,cols,ch = img.shape
# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
# M = cv.getPerspectiveTransform(pts1,pts2)
# dst = cv.warpPerspective(img,M,(300,300))
# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()