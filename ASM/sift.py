import cv2.cv as cv
import cv2
import numpy as np  
# import pdb  
# pdb.set_trace()#turn on the pdb prompt  
  
#read image
def packageKeyPoint(x,y):
    # mpt=(float(x),float(y))
    mkpt=cv2.KeyPoint(float(x),float(y),2)
    return mkpt

if __name__=="__main__":
    img = cv2.imread('lena.png',cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    # cv2.imshow('origin',img);

    # s=packageKeyPoint(1,2)
    #SIFT  
    detector = cv2.SIFT()

    keypoints = detector.detect(gray,None)
    # print type(keypoints)
    kp=keypoints[100]

    print kp.angle
    print kp.class_id
    print kp.size   
    print kp.octave
    print kp.pt
    kp=[cv2.KeyPoint(19.0,20.0,2),cv2.KeyPoint(19.0,120.0,6)]
    keypoints,des = detector.compute(gray,kp)
    print des
    # print len(keypoints)
    # print type(des)
    # print des.shape

    # img = cv2.drawKeypoints(gray,keypoints)  
    # #img = cv2.drawKeypoints(gray,keypoints,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  
    # cv2.imshow('test',img);  
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()  