import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
irisfile = pd.read_csv('iris.csv.txt')
import tools_4211 as ts
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

irisarray = irisfile.to_numpy()


data = irisarray[:, :2]
labels = irisarray[:, -1]
print(data)
print(labels)

cols = np.array([[1,0,0],[0,1,0],[0,0,1]])
ts.plot_2D_data(data,labels,colours = cols, msize = 5)
plt.show()

trd, tsd, trl, tsl = train_test_split(data,labels, test_size=0.5)

cla1 = LinearDiscriminantAnalysis()
cla1.fit(trd, trl)
assigned_labels = cla1.predict(tsd)
error_rate = np.mean(tsl != assigned_labels)
print(error_rate)


cla2 = KNeighborsClassifier()
cla2.fit(trd, trl)
assigned_labels = cla2.predict(tsd)
error_rate = np.mean(tsl != assigned_labels)
print(error_rate)

cla3 = DecisionTreeClassifier()
cla3.fit(trd, trl)
assigned_labels = cla3.predict(tsd)
error_rate = np.mean(tsl != assigned_labels)
print(error_rate)

cols = np.array([[1,0,0],[0,1,0],[0,0,1]])
#ts.plot_2D_data(cla1,cla2,cla3,colours = cols, msize = 5)
#plt.show()





