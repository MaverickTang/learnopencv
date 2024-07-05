#%% 从相机中读取视频
import numpy as np
import cv2 as cv
# 创建一个 VideoCapture 对象,参数可以是设备索引或视频文件的名称。设备索引就是指定哪个摄像头的数字
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
        # cap.get(0)#访问该视频的某些功能，其中propId是0到18之间的一个数字。
        # 每个数字表示视频的属性（如果适用于该视频），并且可以显示完整的详细信息在这里看到：cv::VideoCapture::get()
        #其中一些值可以使用cap.set(propId，value)进行修改
    # 逐帧捕获
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True，检查此返回值来检查视频的结尾
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 我们在框架上的操作到这里
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 显示结果帧e
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()



#%% 从文件播放视频
import numpy as np
import cv2 as cv
cap = cv.VideoCapture('vtest.avi')
while cap.isOpened():
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
    #使用适当的时间cv.waitKey()。如果太小，则视频将非常快，而如果太大，则视频将变得很慢
        break
cap.release()
cv.destroyAllWindows()

#%%保存视频
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
# 定义编解码器并创建VideoWriter对象
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 0)
    # 写翻转的框架
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# 完成工作后释放所有内容
cap.release()
out.release()
cv.destroyAllWindows()