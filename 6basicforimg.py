import numpy as np
import cv2 as cv
img=cv.imread('cyber2.jpg')

## 获取图像属性
# 获取图像形状
print( img.shape )
# 获取像素总数
print( img.size )
# 图像数据类型
print( img.dtype ) #非常重要！！！

## 可以通过行和列坐标来访问像素值
px = img[100,100]
print(px)
blue=img[100,100,0]
print(blue)
# 可以使用numpy来做单个像素访问
# 访问 RED 值
print(img.item(10,10,2))
# 修改 RED 值
img.itemset((10,10,2),100)


# 展示图片
cv.imshow('image',img)
k = cv.waitKey(0)
if k == 27:         # 等待ESC退出
    cv.destroyAllWindows()

