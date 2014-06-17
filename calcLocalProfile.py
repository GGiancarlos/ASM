#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
#Calculate the local profile at the keypoint aligned.
#Here we use sift-descriptor.
import cv2
import sys
import os
from align import *
import numpy as np
global INFINITE_GRADIENT
INFINITE_GRADIENT=1000
def getAngle(ptA,ptB):
	# print ptA,ptB
	if ptA[0]==ptB[0]:
		return 0
	else:
		tg=(ptA[1]-ptB[1])/(ptA[0]-ptB[0])
		return np.arctan(tg)*180.0/np.pi+90
def calcSiftDes(imgName,points):
	img=cv2.imread(imgName,cv2.IMREAD_COLOR)
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	(height,width,channel)=img.shape
	cnt=len(points)
	kp=[]
	for i in range(cnt/2):
		points[2*i]=0.5*width+points[2*i]
		points[2*i+1]=0.5*height-points[2*i+1]

		kp.append(cv2.KeyPoint(points[2*i],points[2*i+1],4))
	for i in range(cnt/2):
		if i==0:
			ptA=kp[cnt/2-1].pt
			ptB=kp[1].pt
		elif i==cnt/2-1:
			ptA=kp[cnt/2-2].pt
			ptB=kp[0].pt
		else:
			ptA=kp[i-1].pt
			ptB=kp[i+1].pt
		kp[i].angle=getAngle(ptA,ptB)
		# print kp[i].angle
	# print kp
	# keypoints = sift.detect(gray,None)
	# print kp
	# kp=[cv2.KeyPoint(343.0,346.5,2),cv2.KeyPoint(249.0,335.0,6)]
	# print kp
	sift=cv2.SIFT()

	kps,des=sift.compute(gray,kp)
	# sz=des.size
	# des.reshape(sz)
	# des.shape=(1,sz)
	# print des.shape
	# img=cv2.drawKeypoints(img,kp)
	# cv2.imshow(imgName,img)
	# cv2.waitKey(0)
	return des

def test(imgName,points):
	img=cv2.imread(imgName)
	(height,width,channel)=img.shape
	cnt=len(points)
	for i in range(cnt/2):
		x=0.5*width+points[2*i]
		y=0.5*height-points[2*i+1]
		cv2.circle(img,(int(x),int(y)),2,(255,0,0,255))
	cv2.imshow(imgName,img)
	cv2.waitKey(0)
	return

if __name__=="__main__":
	argvs=sys.argv
	CAMERA_LIST=('a','b','c','d','e')

	camera='a'
	FLAG=True
	if len(argvs)==3:
		camera=argvs[2]
		SAVEPATH="muct-landmarks-v1\muct-"+camera+"-jpg-v1\jpg"
		if not camera in CAMERA_LIST:
			FLAG=False
	else:
		print "Usage:\ncalcLocalProfile.py -camera [camera]"
		print "default: camera=a"
		# print np.arctan(1)*4
		FLAG=False
	if FLAG:
		print "Calculate Local Profile,YuliWANG@SunYatSenUniv.\nRunning..."

		cnt=0
		for root,dirs,fn in os.walk(SAVEPATH):
			cnt=len(fn)
			print cnt-3,"images found."
			print "model: ",fn[cnt-2],fn[cnt-1]
		absolutePath=os.getcwd()+"\\"+root+"\\"
		FILE=absolutePath+fn[cnt-2]
		pcaMatrix,meanShape,alignedSet=getDataFromModel(FILE)
		# print pcaMatrix
		# print meanShape
		# print len(alignedSet)
		DESNAME=absolutePath+"muct-"+camera+".profile"
		fout=open(DESNAME,"w")
		fout.writelines("LocalProfile caclulated with SIFT descriptor.")
		fout.writelines("Points per Image: "+str(meanShape.size/2)+str("\n"))
		for i in range(cnt-3):
			imgName=absolutePath+fn[i]
			print imgName," Loaded"
			des=calcSiftDes(imgName,alignedSet[i])
			fout.writelines(str(i)+":"+str(des.tolist()))
			fout.writelines("\n")