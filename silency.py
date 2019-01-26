import time
import cv2
import numpy as np
from utils.resizeimage import Reformat_Image

IMAGEHEIGHT,IMAGEWIDTH = 128,128

def Processimage(frame):
	''' Process the image and extracts the Reason Of Intrest(ROI)
	
	Parameters:
		frame of the video input
	
	Results:
		ROI: Region of interest
		box: bounding box of ROI relative to frame(image) size.

	'''
	frameh, framew = frame.shape[0], frame.shape[1]
	saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	#saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
	(success, saliencyMap) = saliency.computeSaliency(frame)
	saliencyMap = (saliencyMap * 255).astype("uint8")
	saliencyMap = cv2.threshold(saliencyMap.astype("uint8"),0, 250,cv2.THRESH_TOZERO | cv2.THRESH_OTSU)[1]
	#saliencyMap = cv2.GaussianBlur(saliencyMap,(5,5),0)

	kernel = np.ones((10,10), np.uint8) 
	saliencyMap = cv2.dilate(saliencyMap, kernel, iterations=1)

	kernel = np.ones((10,10), np.uint8) 
	saliencyMap = cv2.erode(saliencyMap, kernel, iterations=1)

	#saliencyMap = cv2.bilateralFilter(saliencyMap,5,150,150)
	#saliencyMap = cv2.threshold(saliencyMap.astype("uint8"), 125, 250,cv2.THRESH_TOZERO | cv2.THRESH_OTSU)[1]
	Contours, Hierarchy = cv2.findContours(image=saliencyMap, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
	#saliencyMap = cv2.drawContours(frame, Contours, -1 , (0,0,255),1)
	ROI = list()
	box = list()
	for cnt in Contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if(h*w<0.001*(frameh*framew) or h/w>3 or w/h>3 or h*w>0.12*(frameh*framew) ):
			continue		
		crop_img = frame[y:y+h, x:x+w].copy()
		img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
		img = Reformat_Image(img,h,w)
		X = (np.array(img, dtype=np.float32)*2-255)/255
		ROI.append(X)
		box.append([x,y,x+w,y+h])

	ROI = np.array(ROI,dtype = np.float32)
	return ROI, box
	
