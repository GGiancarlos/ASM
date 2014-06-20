#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
import numpy as np
import matplotlib.pyplot as plt

COLOR=["red","black","yellow","green","orange","purple","blue"]
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


if __name__=="__main__":
	print "affineTransfer"
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
