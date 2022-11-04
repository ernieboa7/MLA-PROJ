import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools_4211 as ts

irisfile= pd.read_csv('iris.csv.txt', header=None)
irisarray = irisfile.to_numpy()

HOGfile = pd.read_csv('DataHOG_PCA20.csv', header=None)
LBPfile = pd.read_csv('DataLBP_PCA20.csv', header=None)
RGBfile = pd.read_csv('DataRGB_PCA20.csv', header=None)
Labelfile= pd.read_csv('Labels.csv', header=None)

HOGarray = HOGfile.to_numpy()
LBParray = LBPfile.to_numpy()
RGBarray = RGBfile.to_numpy()
Labelarray= Labelfile.to_numpy()


HOGdata = HOGarray[:, :2]
LBPdata = LBParray[:, :2]
RGBdata = RGBarray[:, :2]
Labeldata = Labelarray[:, -1]


plt.figure()
plt.scatter(HOGdata[:, 0], HOGdata[:, 1], c=Labeldata)
 
plt.figure()
plt.scatter(LBPdata[:, 0], LBPdata[:, 1], c=Labeldata)

plt.figure()
plt.scatter(RGBdata[:, 0], RGBdata[:, 1], c=Labeldata)        
        
plt.show()

     
# 1C
error, predicted_labels, classifier = ts.train_test_ldc(HOGdata,Labeldata,HOGdata,Labeldata)