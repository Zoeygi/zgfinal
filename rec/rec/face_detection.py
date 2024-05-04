
# 单独测试图片的人脸

import cv2 as cv
import numpy as np
import os

#当前文件所在文件夹
folder_path_cur=os.path.dirname(os.path.abspath(__file__))

# def face_detect_demo(image):
#     gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY) #在灰度图像基础上实现的
#     face_detector = cv.CascadeClassifier(os.path.join(folder_path_cur,"haarcascade_frontalface_alt2.xml"))  #级联检测器获取文件
#     faces = face_detector.detectMultiScale(gray,1.01,3)    #在多个尺度空间上进行人脸检测
#     #第一个参数是灰度图像
#     #第二个参数是尺度变换，就是向上或者向下每次是原来的多少倍，这里是1.02倍
#     #第三个参数是人脸检测次数，设置越高，误检率越低，但是对于迷糊图片，我们设置越高，越不易检测出来，要试单降低
#     # for x,y,w,h in faces:
#         # cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
#     x,y,w,h =faces[0]
#     cv.namedWindow("labelled",0)
#     image2=image[y:y+h,x:x+w]
#     cv.imshow("labelled",image2)
#     # cv.imwrite(os.path.join(folder_path_cur,"labelled_all.jpg"), image2,[int(cv.IMWRITE_PNG_COMPRESSION), 0])
# #
# #
# #
# src = cv.imread(os.path.join(folder_path_cur,"all.jpg"))  #读取图片

# cv.namedWindow("input image",0)    
# cv.imshow("input image",src)    
# p=cv.Canny(src,200,200)
# face_detect_demo(src)
# cv.imshow("p image",p)


# cv.waitKey(0)   #等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
# cv.destroyAllWindows()  #销毁所有窗口




# def face_detect_demo(image):
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     face_detector = cv.CascadeClassifier(os.path.join(folder_path_cur,"haarcascade_frontalface_alt.xml"))
#     faces = face_detector.detectMultiScale(gray, 1.2, 6)
#     for x, y, w, h in faces:
#         cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
#         print(w,h)
#     cv.imshow("result", image)

# print("--------- Python OpenCV Tutorial ---------")

# capture = cv.VideoCapture(0)
# cv.namedWindow("result", cv.WINDOW_AUTOSIZE)
# while(True):
#     ret, frame = capture.read()
#     frame = cv.flip(frame, 1)#左右翻转
#     face_detect_demo(frame)
#     c = cv.waitKey(10)
#     if c == 27: # ESC
#         break
# cv.waitKey(0)
# cv.destroyAllWindows()


# def readtest():
#     videoname = 'videoname.avi'
#     capture = cv.VideoCapture(videoname )
#     if capture.isOpened():
#         while True:
#             ret,img=capture.read() # img 就是一帧图片            
#             # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
#             if not ret:break # 当获取完最后一帧就结束
#     else:
#         print('视频打开失败！')


import cv2
import mediapipe as mp

# 初始化mediapipe的人脸检测模块
mp_face_detection = mp.solutions.face_detection
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# 加载图像
image_path = os.path.join(folder_path_cur,"all.jpg")
image = cv2.imread(image_path)
src=image

# 将图像转换为RGB格式（mediapipe需要的格式）
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 运行人脸检测
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(rgb_image)

    # 绘制矩形框
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

# 保存绘制了矩形框的图像
# cv2.imwrite("output_image.jpg", image)


cv.namedWindow("input image",0)    
cv.imshow("input image",src)    
cv.imshow("result", image)
# p=cv.Canny(src,200,200)
cv.waitKey(0)   #等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
cv.destroyAllWindows()  #销毁所有窗口