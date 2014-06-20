import numpy as np

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
			if 1.0-traceK/float(traceS)<0.0001:
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