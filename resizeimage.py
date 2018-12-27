import glob
import cv2
import numpy as np

IMAGEHEIGHT,IMAGEWIDTH = 128,128

def Reformat_Image(image,h,w):
    width = w
    height = h
    if(width != height):
        bigside = width if width > height else height
        background = np.zeros((bigside, bigside),dtype = np.uint8)+255
        offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2),0)))
        background[offset[1]:h+offset[1],offset[0]:w+offset[0]] = image.copy()
        image = background.copy()
    image = cv2.resize(image,(IMAGEHEIGHT,IMAGEWIDTH))
    return image

        