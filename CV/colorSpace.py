import cv2 
import numpy as np

origin = cv2.imread('test.jpg')
cv2.namedWindow('origin',cv2.WINDOW_NORMAL)
# BGR to HSV
hsv = cv2.cvtColor(origin,cv2.COLOR_BGR2HSV)

# 定义颜色上下界
lower = np.array([10,50,50])
upper = np.array([90,255,255])

# 获得指定范围内颜色的图像
mask = cv2.inRange(hsv,lower,upper)

res = cv2.bitwise_and(origin,origin,mask=mask)
cv2.imshow('origin',res)

# BGR颜色转换为HSV颜色
green = np.uint8([[[0,255,0]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print(hsv_green)

cv2.waitKey(0)
cv2.destroyAllWindows()
