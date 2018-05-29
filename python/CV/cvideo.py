import cv2
import numpy 
import matplotlib.pyplot as plot 

def video_test():
    # 读取视频文件
    cap = cv2.VideoCapture('./horses.mp4')

    # 定义codec和视频写入对象(fps,size)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc,20.0,(480,720))

    while(cap.isOpened()):
        # 获取帧frame
        ret,frame = cap.read()
        # 帧处理
        #print(frame.shape)
        if ret == True :
            img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            img_gb = cv2.GaussianBlur(img_gray,(5,5),0)
            edges = cv2.Canny(img_gb,80,150)
            flip = cv2.flip(edges,1)
            out.write(edges)
            # cv2.imshow('frame',flip)
            if cv2.waitKey(50) & 0xFF == ord('q') :
                break
        else : break 
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_test()