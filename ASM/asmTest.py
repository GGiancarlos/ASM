#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:2014.6.24
#*********************
from ASMModel import *
import calcLocalProfile as PFC


if __name__=="__main__":
	print "ASMModel Running..."
	camera='a'
	camera_b='b'
	camera_c='c'
	camera_d='d'
	camera_e='e'
	DATAPATH="muct-landmarks-v1\muct-"+camera_b+"-jpg-v1\jpg"
	for root,dirs,fn in os.walk(DATAPATH):
			cnt=len(fn)
	absolutePath=os.getcwd()+"\\"+root+"\\"
	MODELNAME="muct-landmarks-v1\muct-"+camera+"-landmarks_original.model"

	ALIGNEDMODELNAME="muct-landmarks-v1\muct-"+camera+"-landmarks_aligned_2.model"
	ALIGNEDMODELNAME_B="muct-landmarks-v1\muct-"+camera_b+"-landmarks_aligned_2.model"
	ALIGNEDMODELNAME_C="muct-landmarks-v1\muct-"+camera_c+"-landmarks_aligned_2.model"
	ALIGNEDMODELNAME_D="muct-landmarks-v1\muct-"+camera_d+"-landmarks_aligned_2.model"
	ALIGNEDMODELNAME_E="muct-landmarks-v1\muct-"+camera_e+"-landmarks_aligned_2.model"


	PROFILENAME="muct-landmarks-v1\muct-"+camera+"_siftWithCanny.profile"
	PROFILENAME_B="muct-landmarks-v1\muct-"+camera_b+"_sift.profile"
	PROFILENAME_C="muct-landmarks-v1\muct-"+camera_c+"_sift.profile"
	PROFILENAME_D="muct-landmarks-v1\muct-"+camera_d+"_sift.profile"
	PROFILENAME_E="muct-landmarks-v1\muct-"+camera_e+"_sift.profile"

	
	IMGNAMEVEC=[]
	for i in range(751):
			IMGNAMEVEC.append(absolutePath+fn[i])
	# asmBuilder=ASM_ModelBuilder(modelName=MODELNAME,saveFileName=PROFILENAME,imgNameVec=IMGNAMEVEC)
	# asmBuilder.setLocalProfileFun(PFC.calcSiftDes)
	# asmBuilder.buildModel()

	
	asmFitter=MultiOrientationASM_Fitter(PROFILENAME,ALIGNEDMODELNAME,CASCADE_LEFT_EYE,CASCADE_FACE_EYE,
										# PROFILENAME_B,ALIGNEDMODELNAME_B,
										# PROFILENAME_C,ALIGNEDMODELNAME_C,
										# PROFILENAME_D,ALIGNEDMODELNAME_D,
										# PROFILENAME_E,ALIGNEDMODELNAME_E,
										)
	asmFitter.setLocalProfileFun(PFC.calcSiftDes)
	asmFitter.setSearchWindow(10)

	################################Test Case#########################
	# t=0
	# img=cv2.imread(IMGNAMEVEC[t])
	# asmFitter.multiFitShape(img)
	# asmFitter.visualizeResult(IMGNAMEVEC[t])


	# t=2
	# img=cv2.imread(IMGNAMEVEC[t])
	# asmFitter.multiFitShape(img)
	# asmFitter.visualizeResult(IMGNAMEVEC[t])
	

	# t=np.random.randint(len(IMGNAMEVEC))
	# img=cv2.imread(IMGNAMEVEC[t])
	# asmFitter.multiFitShape(img)
	# asmFitter.visualizeResult(IMGNAMEVEC[t])
	


	# t=np.random.randint(len(IMGNAMEVEC))
	# img=cv2.imread(IMGNAMEVEC[t])
	# asmFitter.multiFitShape(img)
	# asmFitter.visualizeResult(IMGNAMEVEC[t])

	img=cv2.imread("BioID-FaceDatabase-V1.2/BioID_0080.pgm")
	asmFitter.multiFitShape(img)
	asmFitter.visualizeResult("BioID-FaceDatabase-V1.2/BioID_0000.pgm")
	print asmFitter.getFittingResult()

	img=cv2.imread("Face_14.jpg")
	asmFitter.multiFitShape(img)
	asmFitter.visualizeResult("Face_14.jpg")

	img=cv2.imread("Face_18.jpg")
	asmFitter.multiFitShape(img)
	asmFitter.visualizeResult("Face_18.jpg")

	img=cv2.imread("Face_24.jpg")
	asmFitter.multiFitShape(img)
	asmFitter.visualizeResult("Face_24.jpg")

	img=cv2.imread("Face_3.jpg")
	asmFitter.multiFitShape(img)
	asmFitter.visualizeResult("Face_3.jpg")
	img=cv2.imread("Face_3.jpg")

	asmFitter.multiFitShape(img)
	asmFitter.visualizeResult("Face_3.jpg")	
	img=cv2.imread("Face_20.jpg")
	asmFitter.multiFitShape(img)
	asmFitter.visualizeResult("Face_20.jpg")


	img=cv2.imread("Face_3.jpg")
	asmFitter.multiFitShape(img)
	asmFitter.visualizeResult("Face_3.jpg")




	img=cv2.imread("xicore.jpg")
	asmFitter.multiFitShape(img)
	asmFitter.visualizeResult("xicore.jpg")




	img=cv2.imread("Face_9.jpg")
	asmFitter.multiFitShape(img)
	asmFitter.visualizeResult("Face_9.jpg")