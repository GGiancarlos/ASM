#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
import os
import re
p = re.compile(r'.\.py')
def getLines(path):
	sum=0
	for root,dirs,fns in os.walk(path):
		# absolutePath=os.getcwd()+"\\"+root+"\\"
		for f in fns:
			if p.search(f):
				fin=open(f,"r")
				sum+=len(fin.readlines())
		for dir in dirs:
			sum+=getLines(path+dir)

	return sum

if __name__=="__main__":
	print "Counter of Code Lines Running...\nYuliWANG@SunYatSenUniv."
	print getLines(os.getcwd())
