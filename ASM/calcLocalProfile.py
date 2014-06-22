#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
#Calculate the local profile at the keypoint aligned.
#Here we use sift-descriptor.
import cv2
import cv2.cv as cv
import sys
import os
from align import *
import numpy as np
IPLIMAGE="<type 'cv2.cv.iplimage'>"
NARRAY="<type 'numpy.ndarray'>"
global INFINITE_GRADIENT
INFINITE_GRADIENT=1000
def getAngle(ptA,ptB):
	# print ptA,ptB
	if ptA[0]==ptB[0]:
		return 0
	else:
		tg=(ptA[1]-ptB[1])/(ptA[0]-ptB[0])
		return np.arctan(tg)*180.0/np.pi+90
def calcSiftDes(img,mpoints,auto_orientation=False,angle=0):
	points=copy.deepcopy(mpoints)
	# img=cv2.imread(imgName,cv2.IMREAD_COLOR)
	
	(height,width,channel)=img.shape
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cv2.equalizeHist(gray,gray)
	cnt=len(points)
	kp=[]
	for i in range(cnt/2):
		points[2*i]=0.5*width+points[2*i]
		points[2*i+1]=0.5*height-points[2*i+1]

		kp.append(cv2.KeyPoint(points[2*i],points[2*i+1],4))
	for i in range(cnt/2):

		if auto_orientation is True:
			if i==0:
				ptA=kp[cnt/2-1].pt
				ptB=kp[1].pt
			elif i==cnt/2-1:
				ptA=kp[cnt/2-2].pt
				ptB=kp[0].pt
			else:
				ptA=kp[i-1].pt
				ptB=kp[i+1].pt
			temp=getAngle(ptA,ptB)
			kp[i].angle=temp
		else:
			kp[i].angle=angle
	sift=cv2.SIFT()
	kp,des=sift.compute(gray,kp)

	return des,kp

# def test(imgName,points):
# 	img=cv2.imread(imgName)
# 	(height,width,channel)=img.shape
# 	cnt=len(points)
# 	for i in range(cnt/2):
# 		x=0.5*width+points[2*i]
# 		y=0.5*height-points[2*i+1]
# 		cv2.circle(img,(int(x),int(y)),2,(255,0,0,255))
# 	cv2.imshow(imgName,img)
# 	cv2.waitKey(0)
# 	return

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
		absolutePath=os.getcwd()+"\\"+root+"\\"
		# FILE=absolutePath+fn[cnt-2]
		FILE=absolutePath+"muct-"+camera+"-landmarks_original.model"
		pcaMatrix,meanShape,alignedSet=getDataFromModel(FILE)
		# print pcaMatrix
		# print meanShape
		# print len(alignedSet)
		DESNAME=absolutePath+"muct-"+camera+"_2.profile"
		fout=open(DESNAME,"w")
		fout.writelines("LocalProfile caclulated with SIFT descriptor.\n")
		profileVec=[]

		for i in range(751):
			imgName=absolutePath+fn[i]
			img=cv2.imread(imgName,cv2.IMREAD_COLOR)
			# print imgName," Loaded"
			des,kp=calcSiftDes(img,alignedSet[i],auto_orientation=False,angle=0)
			profileVec.append(des)
		mean=sum(profileVec)/751.0
		(mP,nP)=mean.shape

		profileTransVec=[[] for x in range(mP)]
		covVec=[]
		for i in range(751):
			for j in range(mP):
				profileTransVec[j].extend(profileVec[i][j].tolist())

		for j in range(mP):
			data=np.array(profileTransVec[j])
			data.reshape(751*nP)
			data.shape=(751,nP)
			covVec.append(np.cov(data.transpose()))
		print len(covVec)
		print covVec[0].shape

		# convVec=[ele-mean for ele in profileVec]
		# sumVec=[np.array([0.0 for i in range(nP*nP)]) for j in range(mP)]
		# for j in range(mP):
		# 	sumVec[j].reshape(nP*nP)
		# 	sumVec[j].shape=(nP,nP)
		# for i in range(751):
		# 	for j in range(mP):
		# 		conv=np.dot(convVec[i][j].transpose(),convVec[i][j])
		# 		sumVec[j]+=conv
		#write mean profile

		mean.reshape(mP,nP)
		mean.shape=(1,mP*nP)
		fout.writelines("ParametersOfProfile(mP,nP)"+":"+str(mP)+" "+str(nP)+"\n")
		fout.writelines("MeanProfile"+":"+str(mean[0].tolist())+"\n")
		for j in range(mP):
			# sumVec[j]/=751.0

			covVec[j].shape=(1,nP*nP)
			fout.writelines("LandMark "+str(j)+":"+str(covVec[j][0].tolist())+"\n")



			# fout.writelines(str(i)+":"+str(des.tolist()))
			# fout.writelines("\n")