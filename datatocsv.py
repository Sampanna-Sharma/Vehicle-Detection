import os
import glob
import csv

csvfile = open('data.csv','w')
writer = csv.writer(csvfile)
writer.writerows([['fileloc','value']])


datasource = ['vehicle-data/2wheelers','vehicle-data/4wheelers',"vehicle-data/notvehicle"]

values = [[0,1,0],[0,0,1],[1,0,0]]
for source,value in zip(datasource,values):
    fnamelist = glob.glob(source+'/*.jpg')
    for fname in fnamelist:
        writer.writerows([[fname,str(value)]])
    



