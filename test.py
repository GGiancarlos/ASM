#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
import numpy as np

A=np.array([[1,2],[3,4]])
B=np.array([[5.9,6],[7,8]])
print "A:\n",A
print "B:\n",B
C=np.dot(A,B)
print "C=A*B:"
print C
print "B=A^-1*C:"
print np.dot(np.linalg.inv(A),C)
print np.linalg.norm(A,2)
print dir(C)