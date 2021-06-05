# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 22:12:14 2021
检测图片内人脸性别及年龄预测
@author: linux
"""

import cv2
import time

# 检测人脸并绘制人脸bounding box
'''
blobFromImage(image[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]]) -> retval
image：这个就是我们将要输入神经网络进行处理或者分类的图片。
scalefactor：当我们将图片减去平均值之后，还可以对剩下的像素值进行一定的尺度缩放，它的默认值是1，如果希望减去平均像素之后的值，全部缩小一半，那么可以将scalefactor设为1/2
size：这个参数是我们神经网络在训练的时候要求输入的图片尺寸。
mean：需要将图片整体减去的平均值，如果我们需要对RGB图片的三个通道分别减去不同的值，那么可以使用3组平均值，如果只使用一组，那么就默认对三个通道减去一样的值。减去平均值（mean）：为了消除同一场景下不同光照的图片，对我们最终的分类或者神经网络的影响，我们常常对图片的R、G、B通道的像素求一个平均值，然后将每个像素值减去我们的平均值，这样就可以得到像素之间的相对值，就可以排除光照的影响。
swapRB：OpenCV中认为我们的图片通道顺序是BGR，但是我平均值假设的顺序是RGB，所以如果需要交换R和G，那么就要使swapRB=true
'''
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]  # 高就是矩阵有多少行
    frameWidth = frameOpencvDnn.shape[1]  # 宽就是矩阵有多少列
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (frameHeight, frameWidth), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()  # 网络进行前向传播，检测人脸
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])  # bounding box 的坐标
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frameOpencvDnn, bboxes


# 网络模型  和  预训练模型
faceProto = r'./models/opencv_face_detector.pbtxt'
faceModel = r'./models/opencv_face_detector_uint8.pb'

ageProto = r'./models/age_deploy.prototxt'
ageModel = r'./models/age_net.caffemodel'

genderProto = r'./models/gender_deploy.prototxt'
genderModel = r'./models/gender_net.caffemodel'

# 模型均值
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# 加载网络
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
# 人脸检测的网络和模型
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# 打开一个视频文件或一张图片或一个摄像头
frame  = cv2.imread(r'./res/pic/3.jpg', cv2.IMREAD_REDUCED_COLOR_2 )#原图1/2显示
padding = 20
num = 0
# Read frame
t = time.time()

frameFace, bboxes = getFaceBox(faceNet, frame)

if not bboxes:
    print("No face Detected, Checking next frame")

for bbox in bboxes:
    # print(bbox)   # 取出box框住的脸部进行检测,返回的是脸部图片
    face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
           max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
    print("=======", type(face), face.shape)  #  <class 'numpy.ndarray'> (166, 154, 3)
    #
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    print("======", type(blob), blob.shape)  # <class 'numpy.ndarray'> (1, 3, 227, 227)
    genderNet.setInput(blob)   # blob输入网络进行性别的检测
    genderPreds = genderNet.forward()   # 性别检测进行前向传播
    print("++++++", type(genderPreds), genderPreds.shape, genderPreds)   # <class 'numpy.ndarray'> (1, 2)  [[9.9999917e-01 8.6268375e-07]]  变化的值
    gender = genderList[genderPreds[0].argmax()]   # 分类  返回性别类型
    # print("Gender Output : {}".format(genderPreds))
    print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    print(agePreds[0].argmax())  # 3
    print("*********", agePreds[0])   #  [4.5557402e-07 1.9009208e-06 2.8783199e-04 9.9841607e-01 1.5261240e-04 1.0924522e-03 1.3928890e-05 3.4708322e-05]
    print("Age Output : {}".format(agePreds))
    print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

    label = "{},{},{}".format(num, gender, age)
    num += 1
    cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
               cv2.LINE_AA)  # putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
cv2.imshow("Age Gender Demo", frameFace)
print("time : {:.3f} ms".format(time.time() - t))
cv2.waitKey(0)

cv2.destroyWindow("Age Gender Demo")