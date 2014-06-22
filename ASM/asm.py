#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
#test for ASM alg.
import cv2
import cv2.cv as cv
from align import *
import numpy as np
from calcLocalProfile import *
from alignment import *
from init import *
import os
import string
IPLIMAGE="<type 'cv2.cv.iplimage'>"
PATH_A="muct-landmarks-v1\muct-a-jpg-v1\jpg"
PATH_B="muct-landmarks-v1\muct-b-jpg-v1\jpg"
PATH_C="muct-landmarks-v1\muct-c-jpg-v1\jpg"
PATH_D="muct-landmarks-v1\muct-d-jpg-v1\jpg"
PATH_E="muct-landmarks-v1\muct-e-jpg-v1\jpg"
CASCADE_LEFT_EYE="haarcascade_mcs_lefteye.xml"
CASCADE_RIGHT_EYE="haarcascade_mcs_righteye.xml"
CASCADE_FACE_EYE="haarcascade_frontalface_alt.xml"
INFINITE=1000000
COLOR=[(255,0,0,255),(0,255,0,255),(0,0,255,255),(0,255,255,255),(255,0,255,255),(255,255,0,255)]
def loadKeyPoint(FILENAME):
	fin=open(FILENAME,"r")
	cnt=0
	for line in fin.readlines():
		cnt+=1
		if cnt==6:
			data=line.strip().split(":")
			meanPt=data[1].split(" ")

	return meanPt
def detect_and_draw(img, cascade):
    # allocate temporary images
    gray = cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale),
			       cv.Round (img.height / image_scale)), 8, 1)

    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    cv.EqualizeHist(small_img, small_img)

    if(cascade):
        t = cv.GetTickCount()
        faces = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
                                     haar_scale, min_neighbors, haar_flags, min_size)
        t = cv.GetTickCount() - t
        print "detection time = %gms" % (t/(cv.GetTickFrequency()*1000.))
        if faces:
            for ((x, y, w, h), n) in faces:
                # the input to cv.HaarDetectObjects was resized, so scale the 
                # bounding box of each face and convert it to two CvPoints
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)

    cv.ShowImage("result", img)

def test():
	print "Test for ASM alg.YuliWANG@SunYatSenUniv.\nRunning..."
	CASCADE_FILE="haarcascade_frontalface_alt.xml"
	cascade = cv.Load(CASCADE_FILE)

	meanWholePt=loadKeyPoint("asm_01.kp")
	meanEyePt=loadKeyPoint("asm_eye_1.kp")
	img=cv2.imread(PATH_A+"\i000sa-fn.jpg")
	(height,width,channel)=img.shape
	landmarkWholeCnt=len(meanWholePt)/2
	landmarkEyeCnt=len(meanEyePt)/2
	# print dir(cv2)
	for i in range(landmarkWholeCnt):
		x=float(meanWholePt[2*i])+width*0.5
		y=height*0.5-float(meanWholePt[2*i+1])
		cv2.circle(img,(int(x),int(y)),2,(255,255,255,255))
	for i in range(landmarkEyeCnt):
		x=float(meanEyePt[2*i])+width*0.5
		y=height*0.5-float(meanEyePt[2*i+1])
		cv2.circle(img,(int(x),int(y)),3,(255,0,255,255))
	cv2.imshow("test",img)
	cv2.waitKey(0)
	return
def getDataFromProfile(profileName):
	fin=open(profileName,"r")
	M=int(fin.readline().strip().split(":")[1])
	average_profile=np.array([0.0 for i in range(M*128)])
	profileVec=[]
	cnt=0
	for line in fin.readlines():
		temp=line.strip().split(":")[1]
		L=len(temp)
		data_temp=temp[1:L-2].split("], ")
		data=[]
		for i in range(len(data_temp)):
			data.extend(data_temp[i][1:].split(", "))
		profile=np.array(np.vectorize(float)(data))
		average_profile+=profile
		profileVec.append(profile)
		cnt+=1

	average_profile/=cnt
	average_profile.reshape(M*128)
	average_profile.shape=(M,128)
	# print average_profile
	return average_profile
def getDataFromProfile_2(profileName):
	fin=open(profileName,"r")
	fin.readline()
	MN=fin.readline().strip().split(":")[1].strip().split(" ")
	mP,nP=int(MN[0]),int(MN[1])
	convVec=[]
	for i in range(mP+1):
		line=fin.readline().strip().split(":")[1]
		l=len(line)
		data=np.vectorize(float)(line[1:l-1].strip().split(","))
		if i==0:
			average_profile=np.array(data)
			average_profile.reshape(mP*nP)
			average_profile.shape=(mP,nP)
		else:
			conv=np.array(data)
			conv.reshape(nP*nP)
			conv.shape=(nP,nP)
			convVec.append(conv)



	return average_profile,convVec
def drawShape(img,modelShape,color=(0,255,0,255)):
	(M,N)=modelShape.shape
	(height,width,channel)=img.shape
	for i in range(N/2):
		# cv2.circle(img,(int(modelShape[i][0]+width*0.5),int(0.5*height-modelShape[i][1])),2,color)
		
		cv2.circle(img,(int(modelShape[0][2*i]+width*0.5),int(0.5*height-modelShape[0][2*i+1])),2,color)
	return

#return a and b
def search(img,kp,wd,profile,sg):
	inv_sg=np.linalg.inv(sg)
	(mP,nP)=profile.shape
	minDst=INFINITE
	pos=-1
	search_kp=[]
	for i in range(int(kp[0])-wd,int(kp[0])+wd+1):
		for j in range(int(kp[1])-wd,int(kp[1])+wd+1):
			search_kp.extend([i,j])

	search_profile,search_kps=calcSiftDes(img,search_kp,auto_orientation=False,angle=0)

	for t in range(len(search_profile)):
		curProfile=search_profile[t]
		diff=curProfile-profile
		distance=np.dot(np.dot(diff,inv_sg),diff.transpose())


		if minDst>distance[0][0]:
			minDst=distance[0][0]
			pos=t

	# print "\n"

	return search_kp[2*pos:2*pos+2]
def updateModelPoints(img,InitShape,average_profile,sgVec,wd):
	initShape=copy.deepcopy(InitShape)
	(mP,nP)=average_profile.shape
	for i in range(mP):
		kp=[initShape[0][2*i],initShape[0][2*i+1]]
		# kp=[initShape[i][0],initShape[i][1]]

		profile=average_profile[i]
		profile.reshape(nP)
		profile.shape=(1,nP)
		newPos=search(img,kp,wd,profile,sgVec[i])

		initShape[0][2*i],initShape[0][2*i+1]=newPos[0],newPos[1]
		# initShape[i][0],initShape[i][1]=newPos[0],newPos[1]
	return initShape
def updateModelParas(img,meanShape,targetShape,pcaMatrix):
	(mPc,nPc)=pcaMatrix.shape
	b=np.array([0.0 for i in range(1*nPc)])
	b.shape=(nPc,1)

	iterCnt=5
	while iterCnt:
		# print b
		iterCnt-=1
		x=meanShape+np.dot(pcaMatrix,b).transpose()

		y=alignTwoShapes(x,targetShape,True)

		# k=1.0/np.dot(y,meanShape.transpose())

		# y*=k
		pre_b=b
		b=np.dot(pcaMatrix.transpose(),(y-meanShape).transpose())
		bDist=np.linalg.norm(pre_b-b,2)
		print "bDist: ",bDist
		if bDist<0.05:
			break

	return y
def match(img,shape,meanShape,average_profile,sgVec,pcaMatrix,wd):
	print "start match."
	iterCnt=5
	pre_shape=shape
	while iterCnt:
		print iterCnt," iterations","remains"
		if iterCnt==10:

			drawShape(img,shape,COLOR[4])
		iterCnt-=1
		targetShape=updateModelPoints(img,shape,average_profile,sgVec,wd)

		# drawShape(img,targetShape,COLOR[iterCnt])
		# shape=alignTwoShapes(meanShape,targetShape,True)
		pre_shape=shape
		shape=updateModelParas(img,meanShape,targetShape,pcaMatrix)
		shapeDistance=np.linalg.norm(pre_shape-shape,2)
		tolerance=0.01*np.linalg.norm(shape,2)
		print "shapeDistance: ",shapeDistance,"tolerance: ",tolerance
		if shapeDistance<tolerance:
			break

	return shape
if __name__=="__main__":
	print "Test for ASM alg.YuliWANG@SunYatSenUniv.\nRunning..."
	# test()
	MODEL_FILE=PATH_A+"\muct-a-landmarks_aligned_2.model"
	PROFILE_FILE=PATH_A+"\muct-a_2.profile"
	left_eye_cascade =cv2.CascadeClassifier(CASCADE_LEFT_EYE)
	right_eye_cascade=cv2.CascadeClassifier(CASCADE_RIGHT_EYE)
	face_cascade=cv2.CascadeClassifier(CASCADE_FACE_EYE)
	average_profile,sgVec=getDataFromProfile_2(PROFILE_FILE)
	pcaMatrix,meanShape,alignedSet=getDataFromModel(MODEL_FILE)
	sz_ms=meanShape.size
	# meanShape.shape=(sz_ms/2,2)
	sz_pm=pcaMatrix.size
	pcaMatrix.shape=(sz_ms,sz_pm/sz_ms)
	print pcaMatrix.shape
	# print np.dot(pcaMatrix.transpose()[1],pcaMatrix.transpose()[2].transpose())
	imgCnt=0
	for root,dirs,fn in os.walk(PATH_A):
		imgCnt=len(fn)-3
	t=5
	t=0
	for t in range(0,2,60):
		t=np.random.randint(imgCnt)

		imgName=PATH_A+"\\"+fn[t]
		imgNameSave=PATH_A+"\\m2_"+fn[t]
		print t," ",imgName
		img=cv2.imread(imgName)

		# img=cv2.imread("xicore.jpg")
		# img=cv2.imread("Face_3.jpg")
		# img=cv2.imread("Face_23.jpg")

		#initialized shape
		# drawShape(img,meanShape,(255,0,0,255))

		initShape=init(img,face_cascade,left_eye_cascade,meanShape)
		# drawShape(img,initShape,COLOR[0])
		# cv2.imshow(imgNameSave,img)

		# targetShape=updateModelPoints(img,initShape,average_profile,sgVec,3)

		# drawShape(img,targetShape,(0,0,255,255))

		# targetShape=updateModelParas(img,meanShape,targetShape,pcaMatrix)
		targetShape=match(img,initShape,meanShape,average_profile,sgVec,pcaMatrix,10)
		drawShape(img,targetShape,COLOR[2])

		#cv2.imshow(imgName,img)
		cv2.imshow(imgNameSave,img)
		cv2.imwrite(imgNameSave,img)
	cv2.waitKey(0)
	# cvCapture=cv2.VideoCapture(0)

	# while cvCapture:
	# 	ret,img=cvCapture.read()

	# 	initShape=init(img,face_cascade,left_eye_cascade,meanShape)
	# 	drawShape(img,initShape,COLOR[0])
	# 	targetShape=match(img,initShape,meanShape,average_profile,sgVec,pcaMatrix,1)
	# 	drawShape(img,targetShape,COLOR[2])
	# 	cv2.imshow("camera",img)
	# 	if cv2.waitKey(5) == 27:
	# 		break





