import cv2 
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./test2.png')
kernel = np.ones((5,5),np.uint8)

# 定义不同的核函数
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
# 形态变化
erosion = cv2.erode(img,kernel_ellipse,iterations=1)
dilation = cv2.dilate(img,kernel,iterations=1)
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel) #类似erode
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel) #类似dilate
outline = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel) #the difference between input image and Opening of the image
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel) #the difference between the closing of the input image and input image

plt.figure()
plt.subplot(3,3,1), plt.title('origin'),plt.imshow(img)
plt.subplot(3,3,2), plt.title('erosion'),plt.imshow(erosion)
plt.subplot(3,3,3), plt.title('dilate'),plt.imshow(dilation)
plt.subplot(3,3,4), plt.title('opening'),plt.imshow(opening)
plt.subplot(3,3,5), plt.title('closing'),plt.imshow(closing)
plt.subplot(3,3,6), plt.title('gradient'),plt.imshow(outline)
plt.subplot(3,3,7), plt.title('tophat'),plt.imshow(tophat)
plt.subplot(3,3,8), plt.title('blackhat'),plt.imshow(blackhat)
plt.show()

cv2.waitKey(0)