# -*- coding : utf-8 -*-
import cv2 as cv

def test():
    img = cv.imread('./test.jpg')
    newImg = img
    # 设定窗口
    cv.namedWindow('Image',cv.WINDOW_NORMAL) 
    cv.imshow('Image',img)

    print(img.shape)
    print(img.dtype)

    width = img.shape[0]
    height = img.shape[1]
    print(width,height)
    cv.putText(newImg,'hello',(100,100),cv.FONT_HERSHEY_COMPLEX,3,(255,255,255),10)
    # 等待IO
    key = cv.waitKey(0)
    if key == 27:
        cv.destroyAllWindows()

    elif key == ord('s'):
        # 保存图像
        cv.imwrite('newtest.png',newImg)
        print("保存完毕!\n")

    elif key == ord('m'):
        # 放大图像
        res = cv.resize(img,(2*width,2*height),interpolation=cv.INTER_CUBIC)
        cv.imwrite('scaleImg.png',res)
        print("保存完毕!\n")

    elif key == ord('e'):
        # 边缘提取
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_gb = cv.GaussianBlur(img_gray,(5,5),0)
        edges = cv.Canny(img_gray,100,200)
        cv.imshow('edge',edges)
        cv.imwrite('edges.png',edges)
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif key == ord('b'):
        # 图像信息二值化
        th=100
        ret,binary1=cv.threshold(img,th,255,cv.THRESH_BINARY_INV)
        ret,binary2=cv.threshold(img,th,255,cv.THRESH_BINARY)
        ret,binary3=cv.threshold(img,th,255,cv.THRESH_TRUNC)
        ret,binary4=cv.threshold(img,th,255,cv.THRESH_TOZERO)
        ret,binary5=cv.threshold(img,th,255,cv.THRESH_TOZERO_INV)
        #ret,binary6=cv.threshold(img,th,255,cv.THRESH_OTSU)
        #ret,binary7=cv.threshold(img,th,255,cv.THRESH_TRIANGLE)

        cv.imshow("THRESH_BINARY_INV",binary1)
        cv.imshow("THRESH_BINARY",binary2)
        cv.imshow("THRESH_TRUNC",binary3)
        cv.imshow("THRESH_TOZERO",binary4)
        cv.imshow("THRESH_TOZERO_INV",binary5)
        #cv.imshow("THRESH_OTSU",binary6)
        #cv.imshow("THRESH_TRIANGLE",binary7)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    test()