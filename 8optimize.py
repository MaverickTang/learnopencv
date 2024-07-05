# 优化
import numpy as np
import cv2 as cv
img=cv.imread('cyber2.jpg')
## 衡量性能
e1 = cv.getTickCount()# 返回从参考事件（如打开机器的那一刻）到调用此函数那一刻之间的时钟周期数
# 你的执行代码
e2 = cv.getTickCount()
time = (e2 - e1)/ cv.getTickFrequency()

## 默认优化
# 打开优化
print(cv.useOptimized())
print(cv.medianBlur(img,49))
# 关闭优化
cv.setUseOptimized(False)
cv.useOptimized()