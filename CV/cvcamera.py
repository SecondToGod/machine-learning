import cv2 as cv
import numpy
import matplotlib as plot
# 摄像头对象
cap = cv.VideoCapture(0)

# 显示
while(1):
    ret,frame = cap.read()
    cv.imshow('capture',frame)
    if(cv.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv.destroyAllWindows()