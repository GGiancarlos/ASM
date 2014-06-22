#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
import numpy as np
A=np.array(np.random.randint(0,100,9))
A.shape=(3,3)
print A
print A[:,0].shape
print A.transpose()[0].transpose().shape
print np.random.rand()*0.7+0.3
A=[1,2,3,4]
for i in range(2):
	print A[2*i:2*i+2]