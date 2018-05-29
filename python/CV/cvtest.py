# -*- coding : utf-8 -*-
import cv2 as cv2
from matplotlib import pyplot as plot

def test():
    img = cv2.imread('./test.jpg')
    newImg = img
    # 设定窗口
    # cv2.namedWindow('Image',cv2.WINDOW_NORMAL) 
    # cv2.imshow('Image',img)

    # 调用matplotlib画图
    plot.imshow(img,cmap='gray',interpolation = 'bicubic')
    plot.xticks([]),plot.yticks([])
    plot.show()
    
    # print(img.shape)
    # print(img.dtype)

    width = img.shape[0]
    height = img.shape[1]
    # print(width,height)

    # 添加文字
    cv2.putText(newImg,'hello',(100,100),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,255),10)

    # 等待IO
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

    elif key == ord('s'):
        # 保存图像
        cv2.imwrite('newtest.png',newImg)
        print("保存完毕!\n")

    elif key == ord('m'):
        # 放大图像
        res = cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('scaleImg.png',res)
        print("保存完毕!\n")

    elif key == ord('e'):
        # 边缘提取
        img_gray = cv2.cv2tColor(img,cv2.COLOR_BGR2GRAY)
        img_gb = cv2.GaussianBlur(img_gray,(5,5),0)
        edges = cv2.Canny(img_gray,100,200)
        cv2.imshow('edge',edges)
        cv2.imwrite('edges.png',edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif key == ord('b'):
        # 图像信息二值化
        th=100
        ret,binary1=cv2.threshold(img,th,255,cv2.THRESH_BINARY_INV)
        ret,binary2=cv2.threshold(img,th,255,cv2.THRESH_BINARY)
        ret,binary3=cv2.threshold(img,th,255,cv2.THRESH_TRUNC)
        ret,binary4=cv2.threshold(img,th,255,cv2.THRESH_TOZERO)
        ret,binary5=cv2.threshold(img,th,255,cv2.THRESH_TOZERO_INV)
        #ret,binary6=cv2.threshold(img,th,255,cv2.THRESH_OTSU)
        #ret,binary7=cv2.threshold(img,th,255,cv2.THRESH_TRIANGLE)

        cv2.imshow("THRESH_BINARY_INV",binary1)
        cv2.imshow("THRESH_BINARY",binary2)
        cv2.imshow("THRESH_TRUNC",binary3)
        cv2.imshow("THRESH_TOZERO",binary4)
        cv2.imshow("THRESH_TOZERO_INV",binary5)
        #cv2.imshow("THRESH_OTSU",binary6)
        #cv2.imshow("THRESH_TRIANGLE",binary7)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # elif key == ord('b'):
    

if __name__ == '__main__':
    test()