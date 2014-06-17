#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:2014/6/9
#*********************
import numpy as np
import matplotlib.pyplot as plt
import copy
import csv as csv

COLOR=["red","black","yellow","green","orange","purple","blue"]

CSVDATA="D:\MyDocument\CS_Related\AndrewNg_MachineLearning\muct-landmarks-v1\muct-landmarks\muct76.csv"
CSVDATA_PART="D:\MyDocument\CS_Related\AndrewNg_MachineLearning\muct-landmarks-v1\muct-landmarks\muct76 - Copy.csv"
def readData(filename):

	return
def affineTransfer(X,xShift,r,theta):
	thetaVec=[np.cos(theta),-1*np.sin(theta),np.sin(theta),np.cos(theta)]
	A=np.array(thetaVec)
	A.shape=(2,2)
	# print A.shape,X.transpose().shape
	Y=r*np.dot(A,X.transpose())
	Y=Y.transpose()
	# print Y.shape
	(M,N)=Y.shape
	# print Y
	for i in range(M):
		# print Y[i].shape,xShift.shape
		for j in range(N):
			Y[i][j]+=xShift[0][j]
	return Y

def runAffineTest():
	N=50
	Range=100
	xX=np.random.randint(0,Range,N)
	xY=np.random.randint(0,Range,N)
	# print xX,xY,dir(xY)

	xData=xX.tolist()
	xData.extend(xY.tolist())
	X=np.array(xData)
	X.shape=(N,2)
	# X=np.array([[0,0],[1,0],[1,1],[0,1]])
	print X.shape
	xShift=np.array([[10,10]])
	Y=affineTransfer(X,xShift,1,np.random.rand()*3.14)
	fig=plt.figure("affineTransfer")
	(MX,NX)=X.shape
	(MY,NY)=Y.shape
	for i in range(MX):
		plt.plot(X[i][0],X[i][1],COLOR[1],marker="o",label="Original Model")

		plt.plot(Y[i][0],Y[i][1],COLOR[4],marker="*",label="Transfered Model")
	plt.xlim(-150,150)
	plt.ylim(-150,150)
	# plt.legend()
	plt.show()
	return
def runASM(trainData):

	# testData=trainData[np.random.randint(len(trainData))]
	testData=sum(trainData)/len(trainData)
	xShift=np.array([[100,100]])
	Y=affineTransfer(testData,xShift,np.random.rand()*1.5,np.random.rand()*3.14)
	(MX,NX)=testData.shape

	plt.figure("ActiveShapeModelAlgorithm")

	plt.plot(testData.transpose()[0],testData.transpose()[1],COLOR[1],marker=".",label="Original Model",linestyle="-")
	plt.plot(Y.transpose()[0],Y.transpose()[1],COLOR[1],marker=".",label="Original Model")
	# for i in range(MX):
	# 	plt.plot(testData[i][0],testData[i][1],COLOR[1],marker=".",label="Original Model")
	# 	plt.plot(Y[i][0],Y[i][1],COLOR[0],marker="*",label="Original Model")

	plt.xlim(-300,300)
	plt.ylim(-300,300)
	plt.grid('on')
	# plt.legend()
	plt.show()

	return
def cenctrilizeShape(sample):
	
	sz=sample.size
	# sample.shape=(sz/2,2)
	(M,N)=sample.shape
	# print sample
	mean=np.array([sample.transpose()[0].sum()/M,sample.transpose()[1].sum()/M])
	# mean.shape=(2,1)
	for i in range(M):
		sample[i]-=mean
	# sample.shape=(M*N,1)
	return sample
def alignTwoShapes(original,target):
	sz_ori=original.size
	sz_tar=target.size
	# original.shape=(sz_ori/2,2)
	# target.shape=(sz_tar/2,2)
	# M=len(dataSet)
	# averShape=sum(dataSet)/M
	# target=averShape
	N=len(original)
	S=[0.0 for x in range(11)]
	for i in range(N):
		S[0]+=original[i][0]
		S[1]+=original[i][1]
		S[2]+=target[i][0]
		S[3]+=target[i][1]
		S[4]+=original[i][0]*original[i][0]
		S[5]+=original[i][1]*original[i][1]
		S[6]+=original[i][0]*original[i][1]
		S[7]+=original[i][0]*target[i][0]
		S[8]+=original[i][1]*target[i][1]
		S[9]+=original[i][0]*target[i][1]
		S[10]+=original[i][1]*target[i][0]
	for i in  range(11):
		S[i]/=N
	# print original
	# print original.transpose()
	# meanOriginal=np.array([original.transpose()[0].sum()/N,original.transpose()[1].sum()/N])
	# meanTarget=np.array([target.transpose()[0].sum()/N,target.transpose()[1].sum()/N])
	# meanOriginal.shape=(1,2)
	# meanTarget.shape=(1,2)
	# # shift=-1*meanOriginal
	# shift=-1*meanOriginal
	# # shift=np.array([S[2]/N,S[3]/N])
	# # shift.shape=(2,1)
	# # shift=target.transpose()
	# # print shift
	# delta=S[4]*S[5]-S[6]*S[6]
	# SA=np.array([S[7],S[10],S[9],S[8]])
	# SB=np.array([S[5],-1*S[6],-1*S[6],S[4]])
	# SA.shape=(2,2)
	# SB.shape=(2,2)

	# rotateMatrix=np.dot(SA,SB)/delta
	# # print rotateMatrix


	# alignedRes=np.dot(rotateMatrix,original.transpose())
	# alignedRes= alignedRes.transpose()
	# # print alignedRes.shape
	# # print shift.shape
	# alignedRes+=shift

	SA=np.array([S[4],S[6],S[0],S[6],S[5],S[1],S[0],S[1],N])
	SB=np.array([S[7],S[9],S[10],S[8],S[2],S[3]])
	SA.shape=(3,3)
	SB.shape=(3,2)
	transMatrix=np.dot(np.linalg.inv(SA),SB)
	rotateMatrix=transMatrix[:2][:].transpose()
	offsetMatix=transMatrix[2:3][:]
	alignedRes=np.dot(rotateMatrix,original.transpose()).transpose()

	for i in range(N):
		alignedRes[i][0]+=offsetMatix[0][0]
		alignedRes[i][1]+=offsetMatix[0][1]
	# original.shape=(sz_ori,1)
	# target.shape=(sz_tar,1)
	# res=alignedRes.reshape(sz_ori)
	# res.shape=(sz_ori,1)
	return alignedRes
def aligenDataSet(trainData):
	curTarget=trainData[0]
	M=len(trainData)
	N=len(trainData[0])
	cnt=0
	Iter=2
	for i in range(M):
		trainData[i]=cenctrilizeShape(trainData[i])
	while Iter:
		print "Iteration Cnt: ",cnt
		preTarget=copy.deepcopy(curTarget)
		curTarget=np.array([0 for x in range(2*N)])
		curTarget.shape=(N,2)
		for i in range(M):
			trainData[i]=alignTwoShapes(trainData[i],preTarget)

			curTarget+=trainData[i]
		curTarget/=M


		diff=curTarget-preTarget
		# print diff
		dist=np.linalg.norm(diff,2)
		print dist
		if dist<3.5:
			break
		# print preTarget,curTarget
		cnt+=1
		Iter-=1
	for i in range(M):
		trainData[i]=trainData[i].reshape(2*N,1)
		trainData[i].shape=(1,2*N)
	return trainData,curTarget
def PCA(trainSet,radixSampleCnt=0,pcaCnt=None):
	print "runnnig PCA..."
	m=len(trainSet)
	n=len(trainSet[0][0])
	Sigma=np.zeros(n*n)
	Sigma.shape=(n,n)
	# print Sigma.shape
	if radixSampleCnt==0:
		radixSampleCnt=m
	for i in range(radixSampleCnt):
		face=trainSet[i]
		# print face.transpose().shape,face.shape
		Sigma+=np.dot(face.transpose(),face)
	Sigma/=float(radixSampleCnt)
	V,S,U=np.linalg.svd(Sigma)
	# print S
	# print "U: ",U.shape,"V: ",V.shape
	# testFace=V[np.random.randint(100)]
	# testFace.shape=(32,32)
	# testFace=testFace.transpose()
	# plt.imshow(testFace)
	# plt.show()

	# return None,None
	k=0
	if not pcaCnt:
		traceS=S.sum()
		traceK=0.0
		while 1000:
			# print traceK
			if 1.0-traceK/float(traceS)<0.001:
				break
			traceK+=S[k]
			k+=1
	# k=100
	else:
		k=pcaCnt
	Ureduce=U[:k]
	# print kU.shape
	zVec=[]
	print Ureduce.shape,trainSet[0].shape
	for i in range(m):
		Z=np.dot(Ureduce,trainSet[i].transpose())
		zVec.append(Z)
	print "PCA finished. Principal Components Number: ",k

	return zVec,Ureduce
if __name__=="__main__":
	print "ASM Alg runnnig...\nBy YuliWANG@SunYatSenUniv."
	reader = csv.reader(file(CSVDATA_PART, 'rb'))
	cnt=0
	trainData=[]
	for line in reader:
		if cnt>0:
			data=np.vectorize(float)(line[2:])
			data.shape=(data.size/2,2)
			trainData.append(data)


		cnt+=1
	M=len(trainData)
	# M=100
	N=len(trainData[0])
	trainData,averageShape=aligenDataSet(trainData[:M])


	zVec,Ureduce=PCA(trainData)
	averageShape.shape=(N,2)
	plt.figure("Alignment")
	ax1=plt.subplot(211)
	ax2=plt.subplot(212)
	# for i in range(4):

	plt.sca(ax1)

	plt.plot(averageShape.transpose()[0],averageShape.transpose()[1],COLOR[1],marker='o',linestyle=' ',label="Mean Shape")
	# plt.xlim(-120,120)
	# plt.ylim(-30,30)

	# plt.legend()
	plt.xlabel("MeanShape")
	plt.sca(ax2)
	K=len(trainData)
	plt.plot(averageShape.transpose()[0],averageShape.transpose()[1],COLOR[3],marker='o',linestyle=' ',label="Mean Shape")
	for i in range(K):
		trainData[i].shape=(N,2)
		if i==0:
			plt.plot(trainData[i].transpose()[0],trainData[i].transpose()[1],COLOR[0],marker="+",linestyle=' ',label="Aligned DataSet.Top "+str(K))
		else:
			plt.plot(trainData[i].transpose()[0],trainData[i].transpose()[1],COLOR[0],marker="+",linestyle=' ')

	# plt.xlim(-120,120)
	plt.xlabel("Aligned DataSet.Top "+str(K))
	# plt.legend()
	plt.show()





