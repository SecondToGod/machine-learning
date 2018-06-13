import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./test.jpg')

# 低通滤波可用来去噪平滑，高通滤波可用来寻找边缘
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
# plt.subplot(1,2,1),plt.imshow(img)
# plt.subplot(1,2,2),plt.imshow(dst)
# plt.show()

# averaging
blur1 = cv2.blur(img,(6,6)) 
# Gaussian (高斯核必须为正而且是奇数)
blur2 = cv2.GaussianBlur(img,(9,9),0)
# median
blur3 = cv2.medianBlur(img,7)
# bilateral (双边滤波，有利于保存边缘，是具有像素强度差异的高斯滤波)
blur4 = cv2.bilateralFilter(img,9,75,75)

plt.subplot(2,2,1),plt.imshow(blur1),plt.title('averge')
plt.subplot(2,2,2),plt.imshow(blur2),plt.title('Gaussian')
plt.subplot(2,2,3),plt.imshow(blur3),plt.title('median')
plt.subplot(2,2,4),plt.imshow(blur4),plt.title('bilateral')
plt.show()
cv2.waitKey(0)