#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
import numpy as np
import copy
import cv2
def getDataFromModel(ModelName):
	fin=open(ModelName,"r")
	fin.readline()
	fin.readline()
	# meanShape=[]
	# pcaMatrix=None
	cnt=0
	alignedSet=[]

	for line in fin.readlines():
		temp=line.strip().split(":")
		label=int(temp[0])
		data=temp[1].split(" ")
		# print data
		if label==1:
			pcaMatrix=np.array(np.vectorize(float)(data))
		elif label==2:
			meanShape=np.array(np.vectorize(float)(data))
		else:
			alignedSet.append(np.vectorize(float)(data))
	szMean=meanShape.size
	szPca=pcaMatrix.size
	pcaMatrix.reshape(szPca)
	pcaMatrix.shape=(szMean,szPca/szMean)
	meanShape.reshape(1,szMean)
	meanShape.shape=(1,szMean)
	# print meanShape.shape,pcaMatrix.shape
	# print pcaMatrix
			
	return pcaMatrix,meanShape,alignedSet
# def getDataFromProfile(profileName):
# 	fin=open(profileName,"r")
# 	M=int(fin.readline().strip().split(":")[1])
# 	average_profile=np.array([0.0 for i in range(M*128)])
# 	profileVec=[]
# 	cnt=0
# 	for line in fin.readlines():
# 		temp=line.strip().split(":")[1]
# 		L=len(temp)
# 		data_temp=temp[1:L-2].split("], ")
# 		data=[]
# 		for i in range(len(data_temp)):
# 			data.extend(data_temp[i][1:].split(", "))
# 		profile=np.array(np.vectorize(float)(data))
# 		average_profile+=profile
# 		profileVec.append(profile)
# 		cnt+=1

# 	average_profile/=cnt
# 	average_profile.reshape(M*128)
# 	average_profile.shape=(M,128)
# 	# print average_profile
# 	return average_profile
def getDataFromProfile(profileName):
	fin=open(profileName,"r")
	fin.readline()
	MN=fin.readline().strip().split(":")[1].strip().split(" ")
	mP,nP,nResolution=int(MN[0]),int(MN[1]),int(MN[2])
	average_profileVec=[None for i in range(nResolution)]
	covVec=[[] for i in range(nResolution)]
	cnt=0
	for Line in fin.readlines():
		line=Line.strip().split(":")[1]
		l=len(line)
		data=np.vectorize(float)(line[1:l-1].strip().split(","))
		i=cnt/(mP+1)
		j=cnt%(mP+1)
		if j==0:
			average_profile=np.array(data)
			average_profile.reshape(mP*nP)
			average_profile.shape=(mP,nP)
			average_profileVec[i]=average_profile
		else:
			cov=np.array(data)
			cov.reshape(nP*nP)
			cov.shape=(nP,nP)
			# convVec.append(conv)
			covVec[i].append(cov)
		cnt+=1
	return average_profileVec,covVec,nResolution
def drawShape(img,modelShape,color=(0,255,0,255)):
	(M,N)=modelShape.shape
	(height,width,channel)=img.shape
	for i in range(N/2):
		# cv2.circle(img,(int(modelShape[i][0]+width*0.5),int(0.5*height-modelShape[i][1])),2,color)
		
		cv2.circle(img,(int(modelShape[0][2*i]+width*0.5),int(0.5*height-modelShape[0][2*i+1])),1,color)
	return

