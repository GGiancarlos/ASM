#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
from ASMModel import *
import calcLocalProfile as PFC


if __name__=="__main__":
	print "ASMModel Running..."
	camera='a'
	DATAPATH="muct-landmarks-v1\muct-"+camera+"-jpg-v1\jpg"
	for root,dirs,fn in os.walk(DATAPATH):
			cnt=len(fn)
	absolutePath=os.getcwd()+"\\"+root+"\\"
	ALIGNEDMODELNAME=absolutePath+"muct-"+camera+"-landmarks_aligned_2.model"
	MODELNAME=absolutePath+"muct-"+camera+"-landmarks_original.model"
	PROFILENAME=absolutePath+"muct-"+camera+"_3.profile"
	IMGNAMEVEC=[]
	for i in range(751):
			IMGNAMEVEC.append(absolutePath+fn[i])
	# asmBuilder=ASM_ModelBuilder(modelName=MODELNAME,saveFileName=PROFILENAME,imgNameVec=IMGNAMEVEC)
	# asmBuilder.setLocalProfileFun(PFC.calcSiftDes)
	# asmBuilder.buildModel()

	
	asmFitter=ASM_Fitter(PROFILENAME,ALIGNEDMODELNAME,CASCADE_LEFT_EYE,CASCADE_FACE_EYE)
	asmFitter.setLocalProfileFun(PFC.calcSiftDes)
	asmFitter.setSearchWindow(10)

	#Test Case#########################
	t=0
	img=cv2.imread(IMGNAMEVEC[t])
	asmFitter.fitShape(img)
	asmFitter.visualizeResult(IMGNAMEVEC[t])


	t=2
	img=cv2.imread(IMGNAMEVEC[t])
	asmFitter.fitShape(img)
	asmFitter.visualizeResult(IMGNAMEVEC[t])
	

	t=np.random.randint(len(IMGNAMEVEC))
	img=cv2.imread(IMGNAMEVEC[t])
	asmFitter.fitShape(img)
	asmFitter.visualizeResult(IMGNAMEVEC[t])
	


	t=np.random.randint(len(IMGNAMEVEC))
	img=cv2.imread(IMGNAMEVEC[t])
	asmFitter.fitShape(img)
	asmFitter.visualizeResult(IMGNAMEVEC[t])
	



	img=cv2.imread("Face_3.jpg")
	asmFitter.fitShape(img)
	asmFitter.visualizeResult("Face_3.jpg")



	img=cv2.imread("xicore.jpg")
	asmFitter.fitShape(img)
	asmFitter.visualizeResult("xicore.jpg")




	img=cv2.imread("Face_9.jpg")
	asmFitter.fitShape(img)
	asmFitter.visualizeResult("Face_9.jpg")