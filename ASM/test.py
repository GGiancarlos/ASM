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
A=[list([1,2,3,4]),list([5,6,7,8])]
print sum(A)