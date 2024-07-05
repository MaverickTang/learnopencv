# 图像的叠加融合
import numpy as np
import cv2 as cv
img=cv.imread('cyber2.jpg')

#您可以通过OpenCV函数cv.add()或仅通过numpy操作res = img1 + img2添加两个图像。两个图像应具有相同的深度和类型，或者第二个图像可以只是一个标量值。
#注意 OpenCV加法和Numpy加法之间有区别。OpenCV加法是饱和运算，而Numpy加法是模运算。


## 融合两张图片（加水印
# 加载两张图片
img1 = cv.imread('cyber2.jpg')
img2 = cv.imread('opencv-logo-white.png')
# 我想把logo放在左上角，所以我创建了ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
# 现在创建logo的掩码，并同时创建其相反掩码
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
# 现在将ROI中logo的区域涂黑
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
# 仅从logo图像中提取logo区域
img2_fg = cv.bitwise_and(img2,img2,mask = mask)
# 将logo放入ROI并修改主图像
dst = cv.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv.imshow('res',img1)
cv.waitKey(0)
cv.destroyAllWindows()

## 混合两张图片，透明叠加
img1 = cv.imread('cyber2.png')
img2 = cv.imread('opencv-logo-white.png')
# 注意，需要两个图片大小相同
dst = cv.addWeighted(img1,0.7,img2,0.3,0)
cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()