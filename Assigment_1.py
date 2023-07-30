import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools_4211 as ts
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tools_4211 import train_test_knn
from tools_4211 import plot_regions


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

#1a.
# Based on the figure, which data representation would you prefer for solving the 
# classification problem and why?

#Comment: Ans 1a. From the plotted figure HOG, HOG data representation is the best for solving
# classification problem because it has more objects and and better distribution.

#1b: Expected in the report: Calculation of the proportions and 
#the error rate of the classifier, as well as a one sentence explanation.

# The largest data is the RGB DATA set
Accuracy=114
error_rate = 1 -(114/316)
print('error_rate=', error_rate)
 
# Question 1.c Function
# Apply the resubstitution protocol to estimate the error rate of LDC for the three data representations. 
# Give a comment on the result
error, predicted_labels, classifier = ts.train_test_ldc(HOGdata,Labeldata,HOGdata,Labeldata)
cla1 = LinearDiscriminantAnalysis()
cla1.fit(HOGdata, Labeldata)
assigned_labels = cla1.predict(HOGdata)
error_rate = np.mean(Labeldata != assigned_labels)
print(error_rate)

error, predicted_labels, classifier = ts.train_test_ldc(LBPdata,Labeldata,LBPdata,Labeldata)
cla1 = LinearDiscriminantAnalysis()
cla1.fit(LBPdata, Labeldata)
assigned_labels = cla1.predict(LBPdata)
error_rate = np.mean(Labeldata != assigned_labels)
print(error_rate)


error, predicted_labels, classifier = ts.train_test_ldc(RGBdata,Labeldata,RGBdata,Labeldata)
cla1 = LinearDiscriminantAnalysis()
cla1.fit(RGBdata, Labeldata)
assigned_labels = cla1.predict(RGBdata)
error_rate = np.mean(Labeldata != assigned_labels)
print(error_rate)

#   1C Comment: Using Resubstitution to estimate the LDC error rate,
# Show that the HOG data has the minimal amounts of error. Training and testing the 
# HOG data LDC is better the other two data sets. 


# 1(d) Apply the hold-out protocol for the LDC explained in part 1(c)on the 
# three data representations and show the classification errors. Give a comment

trdHOG, tsdHOG, trl, tsl = train_test_split(HOGdata,Labeldata, test_size=0.5)
cla1 = LinearDiscriminantAnalysis()
cla1.fit(trdHOG, trl)
assigned_labels = cla1.predict(tsdHOG)
error_rate = np.mean(tsl != assigned_labels)
print(error_rate)


trdLBP, tsdLBP, trl, tsl = train_test_split(LBPdata,Labeldata, test_size=0.5)
cla1 = LinearDiscriminantAnalysis()
cla1.fit(trdLBP, trl)
assigned_labels = cla1.predict(tsdLBP)
error_rate = np.mean(tsl != assigned_labels)
print(error_rate)


trdRGB, tsdRGB, trl, tsl = train_test_split(RGBdata,Labeldata, test_size=0.5)
cla1 = LinearDiscriminantAnalysis()
cla1.fit(trdRGB, trl)
assigned_labels = cla1.predict(tsdRGB)
error_rate = np.mean(tsl != assigned_labels)
print(error_rate)

# Comment for 1D: Using Hold-out protocol for LDC also shows that the HOG data is best for
# training and testing. The error rate for HOG data set is better than that for the 
# other two data sets.

# Question 1(E)
# OneR classifier for HOG data

def oneR_HOG(t, x):
    if x<t:
        return 3
    elif x>=t:
        return 2
 
t= -0.01383; x=1    
print('HOG_function=', oneR_HOG(-0.01383, 1)) 
clssification_errorHOG= 1 - (102/(316))
print('classification_errorHOG=', clssification_errorHOG)
# the function will return 2 which is the right label
# The classification error will now be calculated as (1-accuracy)


# OneR classifier for LBP data
def oneR_LBP(t, x):
    if x<t:
        return 3
    elif x>=t:
        return 1
 
t= 0.00408; x=2   
print('LBP_function=', oneR_LBP(0.00408, 2)) 
# the function will return 1 which is the right label
# The classification error will now be calculated as (1-accuracy)
clssification_errorLDP= 1 - (100/(316))
print('classification_errorLDP=', clssification_errorLDP)

# OneR classifier for LBP data
Label_left = 3; Label_right = 1
def oneR_RGB(t, x):
    if x<t:
        return 3
    elif x>=t:
        return 1
 
t= 22.42425; x=4   
print('RGB_function=', oneR_RGB(22.42425, 4)) 
# the function will return 3 which is the right label
# The classification error will now be calculated as (1-accuracy)
classification_errorRGB= 1 - (114/(316))
print('classification_errorRGB=', classification_errorRGB)

#Comments for 1D:
# Using oneR classifier to calculate the accuracy of the different data set.
# The RGB data has the best accuracy and the error can be calculated by substracting
# the accuracy from classification_errorRGB= 0.639


# Question 1(f):
  
testing_errors, assigned_labels, cla2= train_test_knn(HOGdata,Labeldata,HOGdata,Labeldata)
cla2 = KNeighborsClassifier()
cla2.fit(HOGdata, Labeldata)
assigned_labels = cla2.predict(HOGdata)
cols = np.array([[1,0,0],[0,1,0],[0,0,1]])
ts.plot_regions(HOGdata, cla2, colours = cols)
plt.show()


# 2. ROC curves
# Suppose that you have a classifier model D, which can be tuned using a parameter 0. You have obtained 
# the following confusion matrices by varying 0:

""" 
      +ve   -ve                    +ve   -ve                 +ve   -ve
+ve    17    783              +ve  92    708            +ve  301   499
-ve     0    800              -ve   1    799            -ve   12   788


      +ve   -ve                    +ve   -ve                 +ve   -ve
+ve   548    252              +ve  723    77            +ve  783    17
-ve    72    728              -ve  227   573            -ve  490   310


     +ve   -ve                    +ve   -ve                 +ve   -ve
+ve  799     1              +ve   800     0           +ve   800     0
-ve  707    93              -ve   780    20           -ve   798     2

"""
    
# The functions for sensitivity and specitivity
def Sensitivity(TP, FN):
    sen_cal = TP/(TP+FN)
    return sen_cal


def Specificity(TN, FP):
    spec_cal = TN/(TN+FP)
    X = (1-spec_cal)
    return spec_cal, X


# (a) To Calculate and plot the ROC curve for D using the parameter of the first model
TP=17; FN=783; FP=0; TN=800
sens1=Sensitivity(TP, FN)
spec1, x1=Specificity(TN, FP)

TP=92; FN=708; FP=1; TN=799
sens2=Sensitivity(TP, FN)
spec2, x2=Specificity(TN, FP)

TP=301; FN=499; FP=12; TN=788
sens3=Sensitivity(TP, FN)
spec3, x3=Specificity(TN, FP)

TP=548; FN=252; FP=72; TN=728
sens4=Sensitivity(TP, FN)
spec4, x4=Specificity(TN, FP)

TP=723; FN=77; FP=227; TN=573
sens5=Sensitivity(TP, FN)
spec5, x5=Specificity(TN, FP)

TP=783; FN=17; FP=490; TN=310
sens6=Sensitivity(TP, FN)
spec6, x6=Specificity(TN, FP)

TP=799; FN=1; FP=707; TN=93
sens7=Sensitivity(TP, FN)
spec7, x7=Specificity(TN, FP)

TP=800; FN=0; FP=780; TN=20
sens8=Sensitivity(TP, FN)
spec8, x8=Specificity(TN, FP)

TP=800; FN=0; FP=798; TN=2
sens9= Sensitivity(TP, FN)
spec9, x9= Specificity(TN, FP)


sensitivity = [sens1, sens2, sens3, sens4, sens5, sens6, sens7, sens8, sens9]
specificity = [spec1, spec2, spec3, spec4, spec5, spec6, spec7, spec8, spec9]

# x is the 1 - specitivity
x = [x1,x2,x3,x4,x5,x6,x7,x8,x9]
print(sensitivity) 
print(specificity)
print(x)
y= sensitivity

 

plt.figure()
plt.grid()
plt.plot(x, y, 'm.-')
plt.plot(0.28374999999999995,0.90375,'k.')
plt.show()        

# 2 (b) Find out which point on the ROC curve is closest to the “ideal” point. 
# Mark on the figure the ideal point and the best operational point. 
# Explain your calculations.

# The ideal point is y=1 and x =0
# From figure 2.a I considered two potential points closect to the ideal point
# And worked out the one with the shortest distance to the ideal point

# closest_distance Z is given as: 
# Z = sqrt((X**2)+(Y**2))
from numpy import sqrt

Z1 = sqrt((0.0899**2) + ((1-0.685)**2))
print('Z1=',Z1)
Z2 = sqrt((0.284**2)+((1-0.903)**2))
print('Z2=',Z2)
 
# From the output Z1 = 0.0.328 and Z2 = 0.3001. The closect point to the ideal point
# is Z2 which has y i.e sensitivity = 0.903 and 1-specificity = 0.284.
# From the plotted diagram it is the fifth matrics which values are:
'''
      +ve    -ve      
+ve   723     77       
-ve   227    573  
           
'''

# (c) Taking the best operational point that you identified in part (b), 
# calculate: sensitivity, specificity, recall, precision, accuracy, 
# and the F1 measure for the classifier. If you are writing your solution in Python,
# use only the numpy library for this task.
def Sensitivity(TP, FN):
    sensitivity = TP/(TP+FN)
    return sensitivity


def Specificity(TN, FP):
    specificity = TN/(TN+FP)
    return specificity

def Recall(TP, FN):
    recall = TP/(TP+FN)
    return recall

def Precision(TP, FP):
    precise = TP/(TP+FP)
    return precise

def Accuracy(TP, TN, FP, FN):
    accurate = (TP+TN)/(TP+TN+FP+FN)
    return accurate

# From the given values the sentivity, specificity, recall, precision and accuracy can be calculated
TP=723; FN=77; FP=227; TN=573
print('Sensitivity=',Sensitivity(TP, FN))
print('Specitivity=',Specificity(TN, FP))
print('Recall=',Recall(TP, FN))
print('Precision=',Precision(TP, FP))
print('Accurancy=',Accuracy(TP, TN, FP, FN))


# To calculate the F1 measure for the classifier
precision = 0.7610526315789473; recall = 0.90375
F1 = 2*(precision*recall)/(precision+recall)
print('F1=',F1)


# Question3: Using NMC Classifier 
# In the figure in Figuure 3. Class 1 is shown with blue triangle and class 2 with red dots. 
# The green plus is a new object, not included in the training data.
 
# 3(a) Write down in a table the data and the labels (dataset Z) 

Z='''
   X1       X2        L
   -2       -2        1
   -1       -2        2
    0        1        2
    1        2        2
    1       -2        1
    2        0        1
    3        0        2
'''
# The value for X1 and X2 for class 1 are represented as X1L1 and X2L1, 
# While X1 and X2 for  class 2 are represented as X1L2 and X2L2
X1L1 = -2, 1, 2;                 X1L2 = -1, 0, 1, 3
X2L1 = -2, -2, 0;                X2L2 = -2, 1, 2, 0


# (b) Train an NMC using Z and label the new object marked with the green plus using 
# your NMC. Show your calculations.

# The means of the all the classes are:
from numpy import mean
x1l1 = mean(X1L1)           
x2l1 = mean(X2L1) 
x1l2 = mean(X1L2)          
x2l2 = mean(X2L2)

print('x1l1=',x1l1)
print('x2l1=',x2l1)
print('x1l2=',x1l2)
print('x2l2=',x2l2)

# The distance between the new point mark with green X in figure 3 from the question
# is calculated using the formula below:
# The new P has the x1 and x2 parameters from the figure as:
x1 = -1; x2 = -1 
x1x2L1= np.sqrt((x1-x1l1)**2 + (x2-x2l1)**2)
x1x2L2= np.sqrt((x1-x1l2)**2 + (x2-x2l2)**2)
print('P1=',x1x2L1)
print('P2=',x1x2L2)

# 3b.From the result P1= 1.37 and P2 = 2.15: This implies that class1 has the shortest distance 
# to the new object.