import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class1 = np.random.multivariate_normal([3, 6], [[8, 0], [0, 1]], size=[20])
class2 = np.random.multivariate_normal([-4, 3], [[2, -2], [-2, 6]], size=[70])
class3=np.concatenate((class1, class2), axis=0)
labels1=np.ravel(np.concatenate((np.ones([20,1]), np.ones([70,1])*2), axis=0))
labels = np.ravel(np.concatenate((np.ones([30000,1]), np.ones([70000,1])*2), axis=0))
test1 = np.random.multivariate_normal([3, 6], [[8, 0], [0, 1]], size=[30000])
test2 = np.random.multivariate_normal([-4, 3], [[2, -2], [-2, 6]], size=[70000])
data = np.concatenate((test1, test2), axis=0)
a= SVC()
a.fit(class3, labels1)
falseLabels=a.predict(data)
print(accuracy_score(labels, falseLabels))


'''colour=['k', 'r']

for j, i in enumerate(class3):
    plt.scatter(i[:,0], i[:,1], c=colour[j])   

plt.grid()
#plt.show()

co= np.array([[8, 0], [0, 1]])
class11 = multivariate_normal.pdf([3, 6], cov=co)
print(class11)
class22 = multivariate_normal.pdf([-4, 3], [[2, -2], [-2, 6]])

labels = np.ravel(np.concatenate((np.ones([3000,1]), np.ones([7000,1])*2), axis=0))
data = np.concatenate((class11, class22))

print(labels)
print(data)'''
