import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

class1 = np.random.multivariate_normal([0, 4], [[1, 1], [1, 9]], size=100)
class2 = np.random.multivariate_normal([-1, 0], [[7, -1], [-1, 2]], size=100)

class3 = np.random.uniform([-2, -3], [1,2], [100,2])

import numpy as np
labels = np.ravel(np.concatenate((np.zeros([100,1]), np.ones([100,1]), np.ones([100,1])*2), axis=0))
data = np.concatenate((class1, class2, class3))

print(labels)


colours = np.array(['k', 'r', 'b'])
markers = np.array(['.', 'x', '^'])
y=[]
for n, i in enumerate(np.unique(labels)):
    y.append(n)
    plt.scatter(data[labels==i,0], data[labels==i,1], c=colours[n], marker=markers[n])
print(y)

plt.grid('on')
plt.xlim(-15, 12)
plt.ylim(-4, 13.5)
plt.show()
'''x=[]
xx= [13, 5, 'a', 7, 9, 4, 12, 'c', 5, 'f']
for a , b in enumerate(xx):
    x.append(b)

print(x)
print(x.index(5))'''

classifiers = [
LinearDiscriminantAnalysis(),
KNeighborsClassifier(n_neighbors= 1),
DecisionTreeClassifier(),
SVC(),
BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
RandomForestClassifier()
]
accuracies = np.zeros([len(classifiers), 10])
cfs = np.zeros([len(classifiers), 3, 3])
print(accuracies)
print(cfs)

classifiers = [
LinearDiscriminantAnalysis(),
KNeighborsClassifier(n_neighbors= 1),
DecisionTreeClassifier(),
SVC(),
BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
RandomForestClassifier()
]
#accuracies = np.zeros([len(classifiers), 10])
#cfs = np.zeros([len(classifiers), 3, 3])
for idx, classifier in enumerate(classifiers):
    accuracies[idx, :] = cross_val_score(classifier, data, labels, cv=10)
    print(accuracies)
    cfs[idx, :, :] = confusion_matrix(labels, cross_val_predict(classifier, data, labels, cv=10))
    print('cfs=',cfs)
mean_accuracies = np.mean(accuracies, axis=1)

best_idx = np.argmax(mean_accuracies)
plot_labels = ['class 1', 'class 2', 'class 3']
sb.heatmap(cfs[best_idx, :, :], xticklabels=plot_labels, yticklabels=plot_labels, cbar=False, annot=True)
plt.show()    
  
