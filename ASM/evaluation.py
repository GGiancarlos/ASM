#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
#Evaluation the performance of My ASM Algorithm
import numpy as np
import os
import ASMModel as asm
import calcLocalProfile as PFC
import cv2
import cv2.cv as cv
import matplotlib.pyplot as plt
import utils as utils
camera='a'
ALIGNEDMODELNAME="muct-landmarks-v1\muct-"+camera+"-landmarks_aligned_2.model"
PROFILENAME="muct-landmarks-v1\muct-"+camera+"_siftWithCanny.profile"
def getHausdorffDistance_1(predicted,expected,w,h):
	preL=np.array([predicted[0][8]+0.5*w,0.5*h-predicted[0][9]])
	preR=np.array([predicted[0][18]+0.5*w,0.5*h-predicted[0][19]])
	expL=np.array([expected[2],expected[3]])
	expR=np.array([expected[0],expected[1]])


	DL=np.sqrt(np.linalg.norm(preL-expL,1))
	DR=np.sqrt(np.linalg.norm(preR-expR,1))
	DC=np.sqrt(np.linalg.norm(expL-expR,1))

	d=float(max(DL,DR))/DC
	return d
def getHausdorffDistance_2(predicted,expected):
	preL=np.array([predicted[0][8],predicted[0][9]])
	preR=np.array([predicted[0][18],predicted[0][19]])
	expL=np.array([expected[8],expected[9]])
	expR=np.array([expected[18],expected[19]])


	DL=np.sqrt(np.linalg.norm(preL-expL,1))
	DR=np.sqrt(np.linalg.norm(preR-expR,1))
	DC=np.sqrt(np.linalg.norm(expL-expR,1))

	d=float(max(DL,DR))/DC
	return d
def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+10, y+10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255,255))
def getExpectPos(posfile):
	fin=open(posfile,'r')
	fin.readline()
	expected=np.vectorize(int)(fin.readline().strip().split("\t"))
	fin.close()
	return expected
def BioIDDataBaseTest(st,ed,record=False):
	DATAPATH="BioID-FaceDatabase-V1.2/"


	for root,dirs,fn in os.walk(DATAPATH):
		cnt=len(fn)
	print fn[:10]
	RANGE=(st,ed)
	if record:
		fout=open("BioID-FaceDatabase-V1 Benchmark Result_.dist",'a')
	asmFitter=asm.MultiOrientationASM_Fitter(PROFILENAME,ALIGNEDMODELNAME,asm.CASCADE_LEFT_EYE,asm.CASCADE_FACE_EYE,
										# PROFILENAME_B,ALIGNEDMODELNAME_B,
										# PROFILENAME_C,ALIGNEDMODELNAME_C,
										# PROFILENAME_D,ALIGNEDMODELNAME_D,
										# PROFILENAME_E,ALIGNEDMODELNAME_E,
										)
	asmFitter.setLocalProfileFun(PFC.calcSiftDes)
	asmFitter.setSearchWindow(10)
	font = cv.InitFont(cv.CV_FONT_HERSHEY_COMPLEX,
                          3, 3, 0.0, 5, cv.CV_AA)
	for i in range(RANGE[0],RANGE[1]):
		imgName=DATAPATH+fn[2*i+1]
		posfileName=DATAPATH+fn[2*i]
		img=cv2.imread(imgName)
		(h,w,channel)=img.shape
		asmFitter.fitShape(img)

		expectPos=getExpectPos(posfileName)
		predictedPos=asmFitter.getFittingResult()
		dist=getHausdorffDistance_1(predictedPos,expectPos,w,h)

		draw_str(img,(0,0),fn[2*i+1])
		draw_str(img,(0,12),"HausdorffDistance: "+str(dist))
		if record:
			fout.writelines(str(i)+":"+str(dist)+"\n")
		# print i," : ",fn[2*i+1],dist

		asmFitter.visualizeResult("BioID-FaceDatabase-V1 Benchmark")
		cv2.waitKey(5)
	if record:
		fout.close()
	return
def MuctDataBaseTest(st,ed,record=False):
	camera='a'
	camera_b='b'
	camera_c='c'
	camera_d='d'
	camera_e='e'
	DATAPATH="muct-landmarks-v1\muct-"+camera+"-jpg-v1\jpg"

	for root,dirs,fn in os.walk(DATAPATH):
			cnt=len(fn)
	absolutePath=os.getcwd()+"\\"+root+"\\"
	MODELNAME="muct-landmarks-v1\muct-"+camera+"-landmarks_original.model"

	ALIGNEDMODELNAME="muct-landmarks-v1\muct-"+camera+"-landmarks_aligned_2.model"
	ALIGNEDMODELNAME_B="muct-landmarks-v1\muct-"+camera_b+"-landmarks_aligned_2.model"
	ALIGNEDMODELNAME_C="muct-landmarks-v1\muct-"+camera_c+"-landmarks_aligned_2.model"
	ALIGNEDMODELNAME_D="muct-landmarks-v1\muct-"+camera_d+"-landmarks_aligned_2.model"
	ALIGNEDMODELNAME_E="muct-landmarks-v1\muct-"+camera_e+"-landmarks_aligned_2.model"

	PROFILENAME="muct-landmarks-v1\muct-"+camera+"_sift_trainset_0-450.profile"
	PROFILENAME_B="muct-landmarks-v1\muct-"+camera_b+"_sift.profile"
	PROFILENAME_C="muct-landmarks-v1\muct-"+camera_c+"_sift.profile"
	PROFILENAME_D="muct-landmarks-v1\muct-"+camera_d+"_sift.profile"
	PROFILENAME_E="muct-landmarks-v1\muct-"+camera_e+"_sift.profile"
	expectedShape=utils.getDataFromModel(MODELNAME)[2]
	
	IMGNAMEVEC=[]
	for i in range(751):
			IMGNAMEVEC.append(absolutePath+fn[i])

	if record:
		fout=open("MUCT-FaceDatabase-V1 Benchmark Result_multiResolution.dist",'a')
	asmFitter=asm.MultiOrientationASM_Fitter(PROFILENAME,ALIGNEDMODELNAME,asm.CASCADE_LEFT_EYE,asm.CASCADE_FACE_EYE,
										PROFILENAME_B,ALIGNEDMODELNAME_B,
										PROFILENAME_C,ALIGNEDMODELNAME_C,
										PROFILENAME_D,ALIGNEDMODELNAME_D,
										PROFILENAME_E,ALIGNEDMODELNAME_E,
										)
	asmFitter.setLocalProfileFun(PFC.calcSiftDes)
	asmFitter.setSearchWindow(8)

	for i in range(st,ed):
		img=cv2.imread(IMGNAMEVEC[i])
		# cv2.imshow("MUCT-FaceDatabase-V1 Benchmark",img)
		asmFitter.multiFitShape(img)
		predicted=asmFitter.getFittingResult()
		expected=expectedShape[i]

		# dist=np.sqrt(np.linalg.norm(predicted-expected,1))/np.sqrt(np.linalg.norm(expected,1))
		dist=getHausdorffDistance_2(predicted,expected)
		# draw_str(img,(0,0),fn[i])
		# draw_str(img,(0,12),"HausdorffDistance: "+str(dist))
		# asmFitter.visualizeResult("MUCT-FaceDatabase-V1 Benchmark")
		# cv2.waitKey(5)
		print i," : ",fn[i],dist
		if record:
			fout.writelines(str(i)+":"+str(dist)+"\n")
	if record:
		fout.close()
	return
def visualizeResult(dat):
	fin=open(dat,"r")
	fin.readline()
	distVec=[]
	indexVec=[]

	for line in fin.readlines():
		temp=np.vectorize(float)(line.strip().split(":"))
		if temp[1]<=1.5:
			distVec.append(float(temp[1]))
			indexVec.append(int(temp[0]))
	totalCnt=len(indexVec)
	print totalCnt

	ZERO_VEC=np.zeros(totalCnt)
	ONE_VEC=np.zeros(totalCnt)+1
	accuracyVec=[]
	boundVec=[]
	distArray=np.array(distVec)
	for k in range(1,5000,1):
		bound=k/(5000.0/0.45)

		result=np.where(distArray<=bound,ONE_VEC,ZERO_VEC)

		accuracy=sum(result)/float(totalCnt)
		boundVec.append(bound)
		accuracyVec.append(accuracy)


	fig=plt.figure(dat)
	ax1=plt.subplot(211)
	ax2=plt.subplot(212)
	plt.sca(ax1)
	plt.title("ASM Benchmark@ylwang")

	plt.plot(indexVec,distVec,color="red",linestyle="",marker="+")
	plt.xlabel("Index")
	plt.ylabel("HausdorffDistance")
	plt.sca(ax2)
	plt.plot(boundVec,accuracyVec,color="blue",linestyle=" ",marker="+")
	plt.xlabel("Acceptance Bound")
	plt.ylabel("Accuracy")
	plt.grid("on")
	plt.legend() 
	plt.show()
	
	return
if __name__=="__main__":
	print "Evaluation the performance of My ASM Algorithm....\nBy YuliWANG@SunYatSenUniv."
	# BioIDDataBaseTest(0,1250,True)


	# MuctDataBaseTest(451,750,True)
	# visualizeResult("MUCT-FaceDatabase-V1 Benchmark Result_.dist")

	visualizeResult("MUCT-FaceDatabase-V1 Benchmark Result_multiResolution.dist")
	
