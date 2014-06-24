#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
#Class for ASM Model
import numpy as np
from utils import *
from init import *
import calcLocalProfile as siftCaculator
import cv2
import os
CASCADE_LEFT_EYE="haarcascade_mcs_lefteye.xml"
CASCADE_RIGHT_EYE="haarcascade_mcs_righteye.xml"
CASCADE_FACE_EYE="haarcascade_frontalface_alt.xml"
INFINITE=1000000
class ASM_ModelBuilder():
	def __init__(self, modelName=None,saveFileName=None,imgNameVec=None):
		self.modelName=modelName
		self.saveFileName=saveFileName
		self.imgNameVec=imgNameVec
		self.imgCnt=len(self.imgNameVec)
		self.fout=open(self.saveFileName,'w')
		self.resolutionScale=[1.0,0.5,0.25]
		self.pcaMatrix,self.meanShape,self.alignedSet=getDataFromModel(self.modelName)
		self.__mPfCalcLocalProfile=None
	def setLocalProfileFun(self,pfCalcLocalProfile):
 		
		self.__mPfCalcLocalProfile=pfCalcLocalProfile

	def getModel(self):
		return self.pcaMatrix,self.meanShape,self.alignedSet
	def buildModel(self):
		if self.__mPfCalcLocalProfile is None:
			print "unkown local profile Caculator!You should define or specify it."
			print "use ::setLocalProfileFun(pfCalcLocalProfile)"
			return
		print "buildModel"
		profileVec=[[] for t in range(len(self.resolutionScale))]

		for index_img in range(len(self.imgNameVec)):
			imgName=self.imgNameVec[index_img]

			img=cv2.imread(imgName)
			(h,w,channel)=img.shape
			for i in range(0,len(self.resolutionScale)):
				scale=self.resolutionScale[i]

				curImg=cv2.resize(img,(int(scale*w),int(scale*h)))
				des,kp=self.__mPfCalcLocalProfile(curImg,self.alignedSet[index_img],auto_orientation=False,angle=0,scale=scale)
				profileVec[i].append(des)
		self.fout.writelines("LocalProfile caclulated with SIFT descriptor.\n")
		print len(profileVec)
		for index_scale in range(len(self.resolutionScale)):
			mean=sum(profileVec[index_scale])/len(self.imgNameVec)
			(mP,nP)=mean.shape
			if index_scale==0:
				self.fout.writelines("ParametersOfProfile(mP,nP,cntResolution)"+":"+str(mP)+" "+str(nP)+" "+str(len(self.resolutionScale))+"\n")

			profileTransVec=[[] for x in range(mP)]
			covVec=[]
			for i in range(0,self.imgCnt):
				for j in range(0,mP):
					profileTransVec[j].extend(profileVec[index_scale][i][j].tolist())

			for j in range(0,mP):
				data=np.array(profileTransVec[j])
				data.reshape(self.imgCnt*nP)
				data.shape=(self.imgCnt,nP)
				covVec.append(np.cov(data.transpose()))
			mean.reshape(mP,nP)
			mean.shape=(1,mP*nP)
			self.fout.writelines("MeanProfile"+":"+str(mean[0].tolist())+"\n")
			for j in range(0,mP):
				# sumVec[j]/=751.0

				covVec[j].shape=(1,nP*nP)
				self.fout.writelines("LandMark "+str(j)+":"+str(covVec[j][0].tolist())+"\n")
			print index_scale
		self.fout.close()


		return
class ASM_Fitter(object):
	"""docstring for ASMMatch"""
	def __init__(self, profileName,alignedModelName,eyeCascade,faceCascade):
		super(ASM_Fitter, self).__init__()
		self.profileName = profileName
		self.alignedModelName=alignedModelName
		self.pcaMatrix,self.meanShape,self.alignedSet=getDataFromModel(self.alignedModelName)
		self.average_profile,self.sgVec,self.nResolution=getDataFromProfile(self.profileName)
		self.eye_cascade=cv2.CascadeClassifier(eyeCascade)
		self.face_cascade=cv2.CascadeClassifier(faceCascade)
		self.__fittingResult=None
		self.__mImg=None
		self.__mPfCalcLocalProfile=None
		self.__mSeachWindowWd=15

	def setSearchWindow(self,wd):
		self.__mSeachWindowWd=wd
	def setLocalProfileFun(self,pfCalcLocalProfile):
 		
		self.__mPfCalcLocalProfile=pfCalcLocalProfile
		return
	def __search(self,img,kp,wd,profile,sg,_scale=1.0):
		inv_sg=np.linalg.inv(sg)
		(mP,nP)=profile.shape
		minDst=INFINITE
		pos=-1
		search_kp=[]
		for i in range(int(kp[0])-wd,int(kp[0])+wd+1):
			for j in range(int(kp[1])-wd,int(kp[1])+wd+1):
				search_kp.extend([i,j])
		search_profile,search_kps=self.__mPfCalcLocalProfile(img,search_kp,auto_orientation=False,angle=0,scale=_scale)

		for t in range(len(search_profile)):
			curProfile=search_profile[t]
			diff=curProfile-profile
			distance=np.dot(np.dot(diff,inv_sg),diff.transpose())


			if minDst>distance[0][0]:
				minDst=distance[0][0]
				pos=t

		# print "\n"

		return search_kp[2*pos:2*pos+2],minDst
	def __updateModelPoints(self,img,InitShape,average_profile,sgVec,wd,scale):
		initShape=copy.deepcopy(InitShape)
		(mP,nP)=average_profile.shape
		sumDist=0.0
		for i in range(mP):
			kp=[initShape[0][2*i],initShape[0][2*i+1]]
			# kp=[initShape[i][0],initShape[i][1]]
	
			profile=average_profile[i]
			profile.reshape(nP)
			profile.shape=(1,nP)
			newPos,curMinDst=self.__search(img,kp,wd,profile,sgVec[i],scale)
			sumDist+=curMinDst
			initShape[0][2*i],initShape[0][2*i+1]=newPos[0],newPos[1]
			# initShape[i][0],initShape[i][1]=newPos[0],newPos[1]
		return initShape,sumDist
	def __updateModelParas(self,img,meanShape,targetShape,pcaMatrix):
		(mPc,nPc)=pcaMatrix.shape
		b=np.array([0.0 for i in range(1*nPc)])
		b.shape=(nPc,1)
	
		iterCnt=8
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
			# print "bDist: ",bDist
			if bDist<0.05:
				break
	
		return y
	def __match(self,img,shape,meanShape,average_profile,sgVec,pcaMatrix,wd,scale):
		iterCnt=5
		while iterCnt:
			# print iterCnt," iterations","remains"
			# if iterCnt==10:
	
			# 	drawShape(img,shape,COLOR[4])
			iterCnt-=1
			targetShape,sumDist=self.__updateModelPoints(img,shape,average_profile,sgVec,wd,scale)
			# print targetShape
			# drawShape(img,targetShape,COLOR[iterCnt])
			# shape=alignTwoShapes(meanShape,targetShape,True)
			pre_shape=shape
			shape=self.__updateModelParas(img,meanShape,targetShape,pcaMatrix)
			shapeDistance=np.linalg.norm(pre_shape-shape,2)
			tolerance=0.01*np.linalg.norm(shape,2)
			# print "shapeDistance: ",shapeDistance,"tolerance: ",tolerance
			if shapeDistance<tolerance:
				break
	
		return shape,sumDist


	def fitShape(self,img):
		if img is None:
			print "invalid img!"
			return
		if self.__mPfCalcLocalProfile is None:
			print "unkown local profile Caculator!You should define or specify it."
			print "use ::setLocalProfileFun(pfCalcLocalProfile)"
			return
		self.__mImg=img
		minProfileDist=1000000
		scale=1.0
		(h,w,channel)=img.shape
		minSumDist=10000000
		bestScale=0.0
		bestShape=None

		for i in range(3):
			curImg=cv2.resize(img,(int(scale*w),int(scale*h)))
			curMeanShape=copy.deepcopy(self.meanShape)*scale
			curPcaMatrix=copy.deepcopy(self.pcaMatrix)*scale
			initShape=init(curImg,self.face_cascade,self.eye_cascade,curMeanShape)
			targetShape,sumDist=self.__match(curImg,initShape,curMeanShape,self.average_profile[i],self.sgVec[i],curPcaMatrix,self.__mSeachWindowWd,1.0)
	
			if sumDist<minSumDist:
				bestScale=scale
				minSumDist=sumDist
				bestShape=targetShape

			print "scale:",scale," ",sumDist
			scale/=float(2.0)
		print "Best Scale: ",bestScale


		self.__fittingResult=bestShape/float(bestScale)

		return
	def visualizeResult(self,title="ASM Fitting Result: ",color=(0,0,255,255)):
		if self.__mImg==None:
			print "image not given."
			return
		if self.__fittingResult==None:
			print "shape not fitted"
			return
		drawShape(self.__mImg,self.__fittingResult,color)
		cv2.imshow(title,self.__mImg)
		cv2.waitKey(0)

		return
	def getFittingResult(self):

		return self.__fittingResult
	def evaluation(self):

		return




		
if __name__=="__main__":
	print "ASMModel Running..."
	# camera='a'
	# DATAPATH="muct-landmarks-v1\muct-"+camera+"-jpg-v1\jpg"
	# for root,dirs,fn in os.walk(DATAPATH):
	# 		cnt=len(fn)
	# absolutePath=os.getcwd()+"\\"+root+"\\"
	# ALIGNEDMODELNAME=absolutePath+"muct-"+camera+"-landmarks_aligned_2.model"
	# MODELNAME=absolutePath+"muct-"+camera+"-landmarks_original.model"
	# PROFILENAME=absolutePath+"muct-"+camera+"_3.profile"
	# IMGNAMEVEC=[]
	# for i in range(751):
	# 		IMGNAMEVEC.append(absolutePath+fn[i])
	# # asmBuilder=ASM_ModelBuilder(modelName=MODELNAME,saveFileName=PROFILENAME,imgNameVec=IMGNAMEVEC)
	# # asmBuilder.setLocalProfileFun(siftCaculator.calcSiftDes)

	# # asmBuilder.buildModel()

	
	# asmFitter=ASM_Fitter(PROFILENAME,ALIGNEDMODELNAME,CASCADE_LEFT_EYE,CASCADE_FACE_EYE)
	# asmFitter.setLocalProfileFun(siftCaculator.calcSiftDes)
	# asmFitter.setSearchWindow(10)

	# #Test Case#########################
	# t=0
	# img=cv2.imread(IMGNAMEVEC[t])
	# asmFitter.fitShape(img)
	# asmFitter.visualizeResult(IMGNAMEVEC[t])


	# t=2
	# img=cv2.imread(IMGNAMEVEC[t])
	# asmFitter.fitShape(img)
	# asmFitter.visualizeResult(IMGNAMEVEC[t])
	

	# t=np.random.randint(len(IMGNAMEVEC))
	# img=cv2.imread(IMGNAMEVEC[t])
	# asmFitter.fitShape(img)
	# asmFitter.visualizeResult(IMGNAMEVEC[t])
	


	# t=np.random.randint(len(IMGNAMEVEC))
	# img=cv2.imread(IMGNAMEVEC[t])
	# asmFitter.fitShape(img)
	# asmFitter.visualizeResult(IMGNAMEVEC[t])
	



	# img=cv2.imread("Face_3.jpg")
	# asmFitter.fitShape(img)
	# asmFitter.visualizeResult("Face_3.jpg")



	# img=cv2.imread("xicore.jpg")
	# asmFitter.fitShape(img)
	# asmFitter.visualizeResult("xicore.jpg")




	# img=cv2.imread("Face_9.jpg")
	# asmFitter.fitShape(img)
	# asmFitter.visualizeResult("Face_9.jpg")








	# left_eye_cascade =cv2.CascadeClassifier(CASCADE_LEFT_EYE)
	# face_cascade=cv2.CascadeClassifier(CASCADE_FACE_EYE)
	# average_profile,sgVec,nResolution=getDataFromProfile(PROFILENAME)
	# pcaMatrix,meanShape,alignedSet=getDataFromModel(ALIGNEDMODELNAME)
	# initShape=init(img,face_cascade,left_eye_cascade,meanShape)
	# targetShape,minProfileDist=match(img,initShape,meanShape,average_profile[0],sgVec[0],pcaMatrix,15,1.0)
	# drawShape(img,targetShape,COLOR[2])
	

	# cv2.namedWindow(IMGNAMEVEC[t],flags=cv.CV_WINDOW_AUTOSIZE)
	# cv2.imshow(IMGNAMEVEC[t],img)
	# cv2.waitKey(0)
