import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier


class1 = np.random.multivariate_normal([0, 0], [[2, -1], [-1, 11]], size=5000)
class2a = np.random.multivariate_normal([1, -1], [[4, 2], [2, 3]], size=2500)
class2b = np.random.multivariate_normal([-2, 1], [[2, -1], [-1, 11]], size=2500)
data2 = np.concatenate((class2a, class2b))

labels = np.ravel(np.concatenate((np.ones([5000,1]), np.ones([5000,1])*2), axis=0))
data1 = np.concatenate((class1, data2))
colours = np.array(['k', 'r'])
#label=['class1', 'class2']
for count, label in enumerate(np.unique(labels)):
   plt.scatter(data1[labels==label,0], data1[labels==label,1], s=1, c=colours[count])
plt.grid('on')
plt.ylabel('y')
plt.xlabel('x')
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.show()


rng = np.random.default_rng()
class1a= rng.choice(class1, 20)
class2c= rng.choice(class2a, 10)
class2d = rng.choice(class2b, 10)
data3 = np.concatenate((class2d, class2c))
labe = np.ravel(np.concatenate((np.ones([20,1]), np.ones([20,1])*2), axis=0))
data4 = np.concatenate((class1a, data3))
print(data4)
#data5=rng.choice(data4, (24))
#label5=rng.choice(labe, (40))
colours = np.array(['k', 'r'])
markers = np.array(['.', 'x'])
for count, label in enumerate(np.unique(labe)):
   plt.subplot(2, 3, 1) 
   plt.scatter(data4[labe==label,0], data4[labe==label,1], c=colours[count], marker=markers[count])

plt.grid('on')
plt.ylim(-10, 10)
plt.xlim(-5, 5)
plt.title("Original")


n_bootstraps=5
bootstr=[]
lab_ar=[]
for i in range(n_bootstraps):
   bootstraps, lab_arr = rng.choice(data4, size=len(data4), replace=True), rng.choice(labe, size=len(labe), replace=True)
   bootstr.append(bootstraps)
   lab_ar.append(lab_arr)
   plt.subplot(2, 3, i+2)
   markers = np.array(['.', 'x'])
   colors = np.array(['k', 'r'])
   for count, label in enumerate(np.unique(lab_arr)):
       
       plt.scatter(bootstraps[lab_arr==label,0], bootstraps[lab_arr==label,1], c=colours[count], marker=markers[count])
   

   
   plt.grid('on')
   plt.ylim(-10, 10)
   plt.xlim(-5, 5)
   plt.title("Bootstrap {}".format(i+1))
plt.show()


print(bootstr, lab_ar)









# question 2


# Train a decision tree on the original dataset
tree_orig = DecisionTreeClassifier()
tree_orig.fit(data4, labe)

# Train a decision tree on 5 bootstrap samples
n_samples = data4.shape[0]
n_bootstraps = 5
trees_bs = []
for i in range(n_bootstraps):
    # Create a bootstrap sample
    X_bs, y_bs = resample(data4, labe, n_samples=n_samples, replace=True)
    # Train a decision tree on the bootstrap sample
    tree_bs = DecisionTreeClassifier()
    tree_bs.fit(X_bs, y_bs)
    trees_bs.append(tree_bs)

# Evaluate the performance of each tree on the original dataset
acc_orig = tree_orig.score(data4, labe)
print("Accuracy of tree on original dataset:", acc_orig)

acc_bs = []
# Evaluate the performance of each tree on each bootstrap sample
for tree_bs in trees_bs:
    acc = tree_bs.score(data4, labe)
    acc_bs.append(acc)

# Calculate the average accuracy of the bootstrap trees
avg_acc_bs = np.mean(acc_bs)
print("Average accuracy of trees on bootstrap samples:", avg_acc_bs)

x_min, x_max = -6, 6
y_min, y_max = -6, 6
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
X_grid = np.c_[xx.ravel(), yy.ravel()]


cmap_rb = plt.cm.colors.ListedColormap(['r', 'k'])
plt.subplot(2, 3, 1)
Z_orig = tree_orig.predict(X_grid)
Z_orig = Z_orig.reshape(xx.shape)
plt.contourf(xx, yy, Z_orig, cmap=cmap_rb)
#plt.scatter(data4[:, 0], data4[:, 1], c=labe, colors='k', cmap=plt.cm.Paired)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Original")

# Create a plot for each ensemble member
for i, tree_bs in enumerate(trees_bs):
    plt.subplot(2, 3, i+2)
    Z_bs = tree_bs.predict(X_grid)
    Z_bs = Z_bs.reshape(xx.shape)
    plt.contourf(xx, yy, Z_bs, cmap=cmap_rb)
    #plt.scatter(data4[:, 0], data4[:, 1], c=labe, colors='k', cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Bootstrap {}".format(i+1))

# Show the plots
plt.tight_layout()
plt.show()









# question 3

data, label = make_classification(n_samples=20000, n_features=20, n_informative=10, n_redundant=5, random_state=50)




data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.2, random_state=50)

lr = LogisticRegression(random_state=42)
lr.fit(data_train, label_train)
lr_pred = lr.predict(data_test)
lr_acc = accuracy_score(label_test, lr_pred)

dt = DecisionTreeClassifier(random_state=50)
dt.fit(data_train, label_train)
dt_pred = dt.predict(data_test)
dt_acc = accuracy_score(label_test, dt_pred)

rf = RandomForestClassifier(random_state=50)
rf.fit(data_train, label_train)
rf_pred = rf.predict(data_test)
rf_acc = accuracy_score(label_test, rf_pred)

knn = KNeighborsClassifier()
knn.fit(data_train, label_train)
knn_pred = knn.predict(data_test)
knn_acc = accuracy_score(label_test, knn_pred)

svm = SVC(random_state=50)
svm.fit(data_train, label_train)
svm_pred = svm.predict(data_test)
svm_acc = accuracy_score(label_test, svm_pred)

print(f"Logistic Regression accuracy: {lr_acc}")
print(f"Decision Tree accuracy: {dt_acc}")
print(f"Random Forest accuracy: {rf_acc}")
print(f"K-Nearest Neighbors accuracy: {knn_acc}")
print(f"SVM accuracy: {svm_acc}")




from sklearn.ensemble import BaggingClassifier

classifiers = [lr, dt, rf, knn, svm]
bag = BaggingClassifier(classifiers)
bag.fit(data_train, label_train)
bagging_pred = bag.predict(data_test)
bagging_acc = accuracy_score(label_test, bagging_pred)
print(f"Bagging accuracy: {bagging_acc}")





