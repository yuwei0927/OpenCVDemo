# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:53:39 2021
图像缩放、平移、旋转、仿射、透视变换
https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
@author: YuWei
"""

import numpy as np
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
# 排除警告信息
import warnings
# matplotlib画图常见参数设置
mpl.rcParams["font.family"] = "SimHei" 
# 设置字体
mpl.rcParams["axes.unicode_minus"]=False 
# 用来正常显示负号
plt.rcParams['font.sans-serif']=['SimHei'] 
# 用来正常显示中文标签# 嵌入式显示图形
#%matplotlib inline
warnings.filterwarnings("ignore")

print("Build with OpenCV " + cv2.__version__)

#读取图片
img = cv2.imread(r'./pic/1.jpg',cv2.IMREAD_GRAYSCALE)
rows,cols = img.shape

#设置旋转角度,np.sin()使用弧度计算
rote = 45
pi_rote = np.pi*45/180

#变换矩阵
n=cols/2
m=rows/2
change_ax=np.matrix([[1,0,0],[0,-1,0],[-n,m,1]])
rote_img=np.matrix([[np.cos(pi_rote),-np.sin(pi_rote),0],[np.sin(pi_rote),np.cos(pi_rote),0],[0,0,1]])
change_back=np.matrix([[1,0,0],[0,-1,0],[n,m,1]])
T1=np.matmul(change_ax,rote_img)
T2=np.matmul(T1,change_back)
T=T2.I

#构建一个同样规格的图片
img1 = np.ones((rows,cols), np.uint8)*255

#利用变换矩阵，算该图片像素对应的灰度
for i in range(cols):
    for j in range(rows):
        rloc=[i,j,1]
        oloc=np.matmul(rloc,T)
        x,y= np.ceil(oloc[0,0]).astype(int), np.ceil(oloc[0,1]).astype(int)
        if (x<0 or x>cols-1) or(y<0 or y>rows-1):
            cor=255
        else:
            cor=img.item(x,y)
            img1.itemset((i,j),cor)
        
#显示变换后的图像
plt.subplot(1,2,1)
plt.title('原始图')
plt.imshow(img)
plt.subplot(1,2,2)
plt.title('旋转45度')
plt.imshow(img1)
plt.show()

cv2.imshow("pic", img)
cv2.imshow("res", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()