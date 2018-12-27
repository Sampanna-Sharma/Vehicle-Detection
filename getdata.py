import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
import numpy as np

IMAGEHEIGHT = 128
IMAGEWIDTH = 128

def gettrainingdata(load = True):
    if(not load):
        DataFrame = pd.read_csv("data.csv")
        df_image = DataFrame["fileloc"]
        df_label = DataFrame["value"]

        Y = list()
        X = list()
        for image_data, label_data in zip(df_image, df_label):
            Y.append(ast.literal_eval(label_data))
            img = cv2.imread(image_data,0)
            img = cv2.resize(img,(IMAGEHEIGHT,IMAGEWIDTH))
            X.append(img)

        Y = np.array(Y,dtype=np.float32)
        X = (np.array(X, dtype=np.float32)*2-255)/255
        np.save('Xdata.npy', X)
        np.save('Ydata.npy', Y)
    else:
        X = np.load('Xdata.npy')
        Y = np.load('Ydata.npy')
    print("Total image data size",X.shape)
    print("Total image data size",Y.shape)
    X_train, X_test, y_train, y_test =  train_test_split(X,Y, test_size = 0.1, shuffle = True)
    return  X_train, X_test, y_train, y_test