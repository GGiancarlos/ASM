#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
from alignment import *
from utils import *
import sys
import matplotlib.pyplot as plt
import copy
import csv as csv
import numpy as np
import copy
from PCA import *
COLOR=["red","black","yellow","green","orange","purple","blue"]

CSVDATA="muct-landmarks-v1\muct-landmarks\muct76.csv"
CSVDATA_EYE="muct-landmarks-v1\muct-landmarks\muct76 - Copy.csv"
def getMarkForCamera(camera,originalData):
	reader = csv.reader(file(originalData, 'rb'))
	cnt=0
	trainData=[]
	for line in reader:
		if (cnt-1)%5==camera-1 and cnt!=0:
			data=np.vectorize(float)(line[2:])
			data.shape=(data.size/2,2)
			trainData.append(data)
		cnt+=1
	return trainData



#.....................................................................................................
if __name__=="__main__":
	argvs=sys.argv
	CAMERA_LIST=('a','b','c','d','e')

	camera='a'
	isAlignNeeded='y'
	FLAG=True
	if len(argvs)==5:
		camera=argvs[2]
		opt=argvs[4]
		SAVEPATH="muct-landmarks-v1\muct-"+camera+"-jpg-v1\jpg"
		if not camera in CAMERA_LIST:
			FLAG=False
	else:
		print "Usage:\nalign.py -camera [camera] -isAlignNeeded [opt]"
		print "default: camera=a isAlignNeeded=y"
		FLAG=False
	if FLAG:
		print "origninal eye landmark data:",CSVDATA_EYE
		if opt=='y':

			saveFileName=SAVEPATH+"\muct-"+camera+"-landmarks_aligned_2"+".model"
			iterCnt=5
		elif opt=='n':
			saveFileName=SAVEPATH+"\muct-"+camera+"-landmarks_original"+".model"
			iterCnt=0
		landMarks=getMarkForCamera(CAMERA_LIST.index(camera)+1,CSVDATA_EYE)
		landMarks,averageShape=aligenDataSet(landMarks,iterCnt)
		zVec,Ureduce=PCA_2(landMarks,pcaCnt=5)
		(pM,pN)=Ureduce.shape
		print "PCA matrix: ",(pM,pN)
		fout=open(saveFileName,"w")
		fout.writelines("format:label:data\n")
		fout.writelines("If label==1,data is the PrincipalComponentsMatrix P.If label==2,data is the mean shape.\n")
		Ureduce=Ureduce.reshape(pM*pN)
		Ureduce.shape=(1,pM*pN)
		for i in range(len(landMarks)+2):
			if i==0:
				data=Ureduce[0].tolist()
			elif i==1:
				data=averageShape[0].tolist()
			else:
				data=landMarks[i-2][0].tolist()
			fout.writelines(str(i+1)+":")
			for j in range(len(data)):
				fout.writelines(str(data[j])+" ")
			fout.write("\n")








