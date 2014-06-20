import numpy as np
def mergeMatrixArray(matrixArr,newShape):
	if matrixArr is None:
		return None
	M=len(matrixArr)
	totalArray=[]
	for i in range(M):
		totalArray.extend(matrixArr[i][0].tolist())
	if newShape[0]*newShape[1]!=len(totalArray):
		print "not valid shape."
		return None
	newMatrix=np.array(totalArray)

	newMatrix.shape=newShape

	return newMatrix
def PCA(trainSet,radixSampleCnt=0,pcaCnt=None):
	print "runnnig PCA..."
	print len(trainSet),trainSet[0].shape
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
	U,S,V=np.linalg.svd(Sigma)
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
			if 1.0-traceK/float(traceS)<0.00001:
				break
			traceK+=S[k]
			k+=1
	# k=100
	else:
		k=pcaCnt
	# Ureduce=U[:k]
	Ureduce=U.transpose()[:k].transpose()
	Ureduce.shape=(n,k)
	# print kU.shape
	zVec=[]
	for i in range(m):
		Z=np.dot(Ureduce.transpose(),trainSet[i].transpose())
		zVec.append(Z)
	print "PCA finished. Principal Components Number: ",k

	return zVec,Ureduce
#using numpy.cov module 
def PCA_2(trainSet,radixSampleCnt=0,pcaCnt=None):
	print "runnnig PCA..."
	
	m=len(trainSet)
	n=len(trainSet[0][0])
	mergedSet=mergeMatrixArray(trainSet,(m,n))
	Sigma=np.cov(mergedSet.transpose())
	U,S,V=np.linalg.svd(Sigma)
	# print S
	print "U: ",U.shape,"V: ",V.shape
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
	Ureduce=U.transpose()[:k].transpose()
	Ureduce.shape=(n,k)
	# print kU.shape
	zVec=[]
	for i in range(m):
		Z=np.dot(Ureduce.transpose(),trainSet[i].transpose())
		zVec.append(Z)
	print "PCA finished. Principal Components Number: ",k

	return zVec,Ureduce
def reconstruction(Z,U):
	m=len(Z)
	xVec=[]
	# print U.shape,U.transpose().shape
	# print t.shape
	# print U.shape
	# print Z[0].shape
	for i in range(m):
		# print Z[i].shape,Z[i].size
		X=np.dot(U,Z[i])
		xVec.append(X)

	return xVec