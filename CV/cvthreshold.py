import cv2
import matplotlib.pyplot as plt
import numpy as np 

img = cv2.imread('./test.jpg',0)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

imgs = [img,thresh1,thresh2,thresh3,thresh4,thresh5]
titles = ['origin','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(imgs[i],'gray')
#     plt.title(titles[i])
# plt.show()

#  自适应二值化
#  加上中值滤波平滑图像
img2 = cv2.medianBlur(img,5)
# cv2.namedWindow('blur',cv2.WINDOW_NORMAL)
# cv2.imshow('blur',img2)
th1 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# plt.subplot(1,3,1),plt.imshow(img2,'gray'),plt.title('medianBlur')
# plt.subplot(1,3,2),plt.imshow(th1,'gray'),plt.title('mean')
# plt.subplot(1,3,3),plt.imshow(th2,'gray'),plt.title('gaussian')
# plt.show()

# otsu 二值化
ret,th3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img3 = cv2.GaussianBlur(img,(5,5),0)
ret,th4 = cv2.threshold(img3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure()
plt.subplot(1,3,1),plt.imshow(img3,'gray'),plt.title('GaussianBlur')
plt.subplot(1,3,2),plt.imshow(th3,'gray'),plt.title('otsu')
plt.subplot(1,3,3),plt.imshow(th4,'gray'),plt.title('gaussian+otsu')
plt.show()
plt.figure()
plt.hist(th4.ravel(),256)
plt.show()
cv2.waitKey(0)