import cv2 as cv
import numpy as np


# 加载彩色灰度图像
img = cv.imread('cyber2.jpg',0)
# 图片展示
# cv.namedWindow('image',cv.WINDOW_NORMAL)#创建一个空窗口，然后再将图像加载到该窗口
#**cv.WINDOW_NORMAL**，则可以调整窗口大小
# 默认情况下，该标志为**cv.WINDOW_AUTOSIZE**
cv.imshow('image',img)
k = cv.waitKey(0)
if k == 27:         # 等待ESC退出
    cv.destroyAllWindows()
elif k == ord('s'): # 等待关键字，保存和退出
    cv.imwrite('cyber2grey.jpg',img)
    cv.destroyAllWindows()#破坏创建的所有窗口,
    # cv.destroyWindow()# 在其中传递确切的窗口名称作为参数