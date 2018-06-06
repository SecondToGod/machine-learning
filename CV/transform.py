import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./test.jpg',cv2.WINDOW_NORMAL)
# cv2.namedWindow('transfrom')
rows,cols = img.shape

# 位移变换
M1 = np.float32([[1,0,50],[0,1,50]])
photo1 = cv2.warpAffine(img,M1,(rows,cols))
# cv2.imshow('transform',photo1)


# 旋转变换
M2 = cv2.getRotationMatrix2D((rows/2,cols/2),180,1)
photo2 = cv2.warpAffine(img,M2,(rows,cols))
# cv2.imshow('rotation',photo2)

# Affine 仿射变换(相对位置关系保留，输入三个点)
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[100,50],[100,250]])
M3 = cv2.getAffineTransform(pts1,pts2)
photo3 = cv2.warpAffine(img,M3,(cols,rows))
#cv2.imshow('Affine',photo3)
plt.subplot(121),plt.imshow(img),plt.title('input')
plt.subplot(122),plt.imshow(photo3),plt.title('output')
plt.show()
cv2.waitKey(0)

# persipective 景深变换(输入四个点位置，任意三个非线性)
pts1 = np.float32([[56,65],[200,53],[28,200],[200,200]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M4 = cv2.getPerspectiveTransform(pts1,pts2)
photo4 = cv2.warpPerspective(img,M4,(rows,cols))
plt.subplot(121),plt.imshow(img),plt.title('input')
plt.subplot(122),plt.imshow(photo4),plt.title('output')
plt.show()
cv2.destroyAllWindows()