import cv2
import numpy as np
import torch
from yolo import yolo
from silency import Processimage
import argparse
 

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--videopath", required=True,
	help="path of video file")
args = vars(ap.parse_args())

model = torch.load("model.tar")

print("model loaded")

IMAGEHEIGHT,IMAGEWIDTH = 128,128

cap = cv2.VideoCapture(args["videopath"])
# loop over frames from the video file stream
i = 0
while True:
    
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (600,400))
    ROI, box= Processimage(frame)
    if (ROI.shape[0] == 0):
        continue
    test_data = torch.from_numpy(ROI).view(-1,1,IMAGEHEIGHT,IMAGEWIDTH)
    Category = (model.forward(test_data).data.cpu().numpy()).reshape(-1,1,3)

    for cat,bbox in zip(Category,box):
        cat = np.exp(cat)/np.sum(np.exp(cat),1)
        ans = np.argmax(cat,1)[0]
        if(ans == 0):
            continue
        elif(ans == 2):
            cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)
            cv2.putText(frame,"4wheelers", (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255,2)
        else:
            cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,255),2)
            cv2.putText(frame,"2wheelers", (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX,0.5, 255,2)


    #cv2.imwrite('videofiles/'+str(i)+'.jpg',frame)
    cv2.imshow("FRAme", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # do a bit of cleanup
cv2.destroyAllWindows()