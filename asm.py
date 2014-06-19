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
def init(img,face_cascade,left_eye_cascade,MeanShape):
	meanShape=copy.deepcopy(MeanShape)
	# drawShape(img,meanShape)
	haar_scale=1.1
	min_neighbors=2
	(height,width,channel)=img.shape
	haar_flags=cv.CV_HAAR_SCALE_IMAGE
	min_facesize=(int(0.3*width),int(0.3*height))
	min_eyesize=(int(0.2*min_facesize[0]),int(0.2*min_facesize[1]))
	max_eyeSize=(int(0.25*min_facesize[0]),int(0.25*min_facesize[1]))
	# (height,width,channel)=img.shape
	gray=cv2.cvtColor(img,cv.CV_BGR2GRAY)
	cv2.equalizeHist(gray, gray)

	leftEye=left_eye_cascade.detectMultiScale(img,haar_scale, min_neighbors, haar_flags,min_eyesize)
	faces=face_cascade.detectMultiScale(img,haar_scale, min_neighbors, haar_flags,min_facesize)
	# facesEye= cv.HaarDetectObjects(gray, right_eye_cascade, cv.CreateMemStorage(0),haar_scale, min_neighbors, haar_flags, (20,20))
	cmp=lambda eye1,eye2:1 if eye1[2]*eye1[3]>eye2[2]*eye2[3] else 0
	EYE_FOUND=1
	# if rightEye:
	# 	for (rx, ry, rw, rh),rn in rightEye:
	# 		image_scale=1
	# 		pt1 = (int(rx * image_scale), int(ry * image_scale))
	# 		pt2 = (int((rx + rw) * image_scale), int((ry + rh) * image_scale))
			# cv.Rectangle(img, pt1, pt2, cv.RGB(0, 255, 0), 3, 8, 0)
	if len(faces):
		faces=faces.tolist()
		faces.sort(cmp)
		for (x, y, w, h) in faces[:1]:

			image_scale=1
			# the input to cv.HaarDetectObjects was resized, so scale the 
			# bounding box of each face and convert it to two CvPoints
			pt1 = (int(x * image_scale), int(y * image_scale))
			pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
			cv2.rectangle(img, pt1, pt2, cv.RGB(0, 0, 255), 3, 8, 0)

			if len(leftEye):
				leftEye=leftEye.tolist()
				EyeVec=[]
				leftEye.sort(cmp)
				# print leftEye
				for (lx, ly, lw, lh) in leftEye[:]:
					# print (x, y, w, h),(lx, ly, lw, lh)
					if (lx+0.5*lw)<x or (lx+0.5*lw)>x+w or (ly+0.5*lh)<y+0.3*h or (ly+0.5*lh)>y+0.5*h:
						continue 
					# print n
					image_scale=1
					# pt1 = (int(lx * image_scale), int(ly * image_scale))
					# pt2 = (int((lx + lw) * image_scale), int((ly + lh) * image_scale))
					# cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)
					EyeVec.append((lx+0.5*lw-0.5*width,0.5*height-(ly+0.5*lh)))
					if len(EyeVec)==2:
						# print EyeVec[0]+EyeVec[1]
						sz_ms=meanShape.size

						meanShape.shape=(sz_ms/2,2)
						m=sz_ms/2
						targetCenter=np.array([(EyeVec[0][0]+EyeVec[1][0])*0.5,(EyeVec[0][1]+EyeVec[1][1])*0.5])
						curCenter=np.array([meanShape.transpose()[0].sum()/m,meanShape.transpose()[1].sum()/m])
						# print meanShape
						for j in range(sz_ms/2):
							meanShape[j]-=(curCenter-targetCenter)
						# print meanShape
						meanShape.shape=(1,sz_ms)

	return meanShape
#return a and b
def search(img,kp,wd,profile,sg):
	inv_sg=np.linalg.inv(sg)
	(mP,nP)=profile.shape
	minDst=INFINITE
	pos=(0,0)
	for i in range(int(kp[0])-wd,int(kp[0])+wd+1):
		for j in range(int(kp[1])-wd,int(kp[1])+wd+1):
			search_kp=[i,j]
			curProfile,search_kp=calcSiftDes(img,search_kp,auto_orientation=False,angle=0)
			diff=curProfile-profile

			distance=np.dot(np.dot(diff,inv_sg),diff.transpose())


			if minDst>distance[0][0]:
				minDst=distance[0][0]
				pos=(i,j)




	# print "\n"

	return pos
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

	iterCnt=10
	while iterCnt:
		iterCnt-=1

		x=meanShape+np.dot(pcaMatrix,b).transpose()
		y=alignTwoShapes(x,targetShape,True)

		k=1.0/np.dot(y,meanShape.transpose())

		y*=k
		pre_b=b
		b=np.dot(pcaMatrix.transpose(),(y-meanShape).transpose())
		diff=np.linalg.norm(pre_b-b,2)
		# print "diff: ",diff
		if diff<0.01:
			break

	return y/k
def match(img,shape,meanShape,average_profile,sgVec,pcaMatrix,wd):

	iterCnt=5
	while iterCnt:
		iterCnt-=1
		targetShape=updateModelPoints(img,shape,average_profile,sgVec,wd)

		# drawShape(img,newShape,COLOR[iterCnt])
		shape=alignTwoShapes(meanShape,targetShape,True)
		# shape=updateModelParas(img,meanShape,targetShape,pcaMatrix)

	return shape
if __name__=="__main__":
	print "Test for ASM alg.YuliWANG@SunYatSenUniv.\nRunning..."
	# test()
	MODEL_FILE=PATH_A+"\muct-a-landmarks_aligned.model"
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
	# t=5
	# t=np.random.randint(imgCnt)
	t=2
	img=cv2.imread(PATH_A+"\\"+fn[t])
	img=cv2.imread("xicore.jpg")
	# img=cv2.imread("Face_3.jpg")
	#initialized shape
	# drawShape(img,meanShape,(255,0,0,255))

	initShape=init(img,face_cascade,left_eye_cascade,meanShape)
	drawShape(img,initShape,COLOR[0])

	# targetShape=updateModelPoints(img,initShape,average_profile,sgVec,3)

	# # drawShape(img,targetShape,(0,0,255,255))

	# targetShape=updateModelParas(img,meanShape,targetShape,pcaMatrix)
	targetShape=match(img,initShape,meanShape,average_profile,sgVec,pcaMatrix,7)
	drawShape(img,targetShape,COLOR[2])


	cv2.imshow("test",img)
	cv2.waitKey(0)

