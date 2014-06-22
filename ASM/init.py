#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
import cv2
import cv2.cv as cv
from align import *
import numpy as np
def init(img,face_cascade,left_eye_cascade,MeanShape):

	meanShape=copy.deepcopy(MeanShape)
	# drawShape(img,meanShape)
	haar_scale=1.1
	min_neighbors=2
	(height,width,channel)=img.shape
	haar_flags=cv.CV_HAAR_SCALE_IMAGE
	min_facesize=(int(0.3*width),int(0.3*height))
	min_eyesize=(int(0.2*min_facesize[0]),int(0.2*min_facesize[1]))
	max_eyeSize=(int(0.25*min_facesize[0]),int(0.25*min_facesize[1]))
	# (height,width,channel)=img.shape
	gray=cv2.cvtColor(img,cv.CV_BGR2GRAY)
	cv2.equalizeHist(gray, gray)

	leftEye=left_eye_cascade.detectMultiScale(img,haar_scale, min_neighbors, haar_flags,min_eyesize)
	faces=face_cascade.detectMultiScale(img,haar_scale, min_neighbors, haar_flags,min_facesize)
	# facesEye= cv.HaarDetectObjects(gray, right_eye_cascade, cv.CreateMemStorage(0),haar_scale, min_neighbors, haar_flags, (20,20))
	cmp=lambda eye1,eye2:1 if eye1[2]*eye1[3]>eye2[2]*eye2[3] else 0
	EYE_FOUND=1
	# if rightEye:
	# 	for (rx, ry, rw, rh),rn in rightEye:
	# 		image_scale=1
	# 		pt1 = (int(rx * image_scale), int(ry * image_scale))
	# 		pt2 = (int((rx + rw) * image_scale), int((ry + rh) * image_scale))
			# cv.Rectangle(img, pt1, pt2, cv.RGB(0, 255, 0), 3, 8, 0)
	if len(faces):
		faces=faces.tolist()
		faces.sort(cmp)
		for (x, y, w, h) in faces[:1]:

			image_scale=1
			# the input to cv.HaarDetectObjects was resized, so scale the 
			# bounding box of each face and convert it to two CvPoints
			pt1 = (int(x * image_scale), int(y * image_scale))
			pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
			cv2.rectangle(img, pt1, pt2, cv.RGB(0, 0, 255), 3, 8, 0)

			if len(leftEye):
				leftEye=leftEye.tolist()
				EyeVec=[]
				leftEye.sort(cmp)
				# print leftEye
				for (lx, ly, lw, lh) in leftEye[:]:
					# print (x, y, w, h),(lx, ly, lw, lh)
					if (lx+0.5*lw)<x or (lx+0.5*lw)>x+w or (ly+0.5*lh)<y+0.3*h or (ly+0.5*lh)>y+0.5*h:
						continue 
					# print n
					image_scale=1
					# pt1 = (int(lx * image_scale), int(ly * image_scale))
					# pt2 = (int((lx + lw) * image_scale), int((ly + lh) * image_scale))
					# cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)
					EyeVec.append((lx+0.5*lw-0.5*width,0.5*height-(ly+0.5*lh)))
					if len(EyeVec)==2:
						# print EyeVec[0]+EyeVec[1]
						sz_ms=meanShape.size

						meanShape.shape=(sz_ms/2,2)
						m=sz_ms/2
						targetCenter=np.array([(EyeVec[0][0]+EyeVec[1][0])*0.5,(EyeVec[0][1]+EyeVec[1][1])*0.5])
						curCenter=np.array([meanShape.transpose()[0].sum()/m,meanShape.transpose()[1].sum()/m])
						# print meanShape
						for j in range(sz_ms/2):
							meanShape[j]-=(curCenter-targetCenter)
						# print meanShape
						meanShape.shape=(1,sz_ms)

	return meanShape