#*********************
#*Author:YuliWANG@SunYatSenUniv.
#*Lang:Python
#*Date:
#*********************
#test for ASM alg.
import cv2
import cv2.cv as cv
PATH_A="muct-landmarks-v1\muct-a-jpg-v1\jpg"
PATH_B="muct-landmarks-v1\muct-b-jpg-v1\jpg"
PATH_C="muct-landmarks-v1\muct-c-jpg-v1\jpg"
PATH_D="muct-landmarks-v1\muct-d-jpg-v1\jpg"
PATH_E="muct-landmarks-v1\muct-e-jpg-v1\jpg"
def loadKeyPoint(FILENAME):
	fin=open(FILENAME,"r")
	cnt=0
	for line in fin.readlines():
		cnt+=1
		if cnt==6:
			data=line.strip().split(":")
			meanPt=data[1].split(" ")

	return meanPt
def detect_and_draw(img, cascade):
    # allocate temporary images
    gray = cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale),
			       cv.Round (img.height / image_scale)), 8, 1)

    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    cv.EqualizeHist(small_img, small_img)

    if(cascade):
        t = cv.GetTickCount()
        faces = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
                                     haar_scale, min_neighbors, haar_flags, min_size)
        t = cv.GetTickCount() - t
        print "detection time = %gms" % (t/(cv.GetTickFrequency()*1000.))
        if faces:
            for ((x, y, w, h), n) in faces:
                # the input to cv.HaarDetectObjects was resized, so scale the 
                # bounding box of each face and convert it to two CvPoints
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)

    cv.ShowImage("result", img)
def test():
	print "Test for ASM alg.YuliWANG@SunYatSenUniv.\nRunning..."
	CASCADE_FILE="haarcascade_frontalface_alt.xml"
	cascade = cv.Load(CASCADE_FILE)

	meanWholePt=loadKeyPoint("asm_01.kp")
	meanEyePt=loadKeyPoint("asm_eye_1.kp")
	img=cv2.imread(PATH_A+"\i000sa-fn.jpg")
	(height,width,channel)=img.shape
	landmarkWholeCnt=len(meanWholePt)/2
	landmarkEyeCnt=len(meanEyePt)/2
	# print dir(cv2)
	for i in range(landmarkWholeCnt):
		x=float(meanWholePt[2*i])+width*0.5
		y=height*0.5-float(meanWholePt[2*i+1])
		cv2.circle(img,(int(x),int(y)),2,(255,255,255,255))
	for i in range(landmarkEyeCnt):
		x=float(meanEyePt[2*i])+width*0.5
		y=height*0.5-float(meanEyePt[2*i+1])
		cv2.circle(img,(int(x),int(y)),3,(255,0,255,255))
	cv2.imshow("test",img)
	cv2.waitKey(0)
	return
if __name__=="__main__":
	print "Test for ASM alg.YuliWANG@SunYatSenUniv.\nRunning..."
	test()
