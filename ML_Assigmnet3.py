import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA



HOGdata = pd.read_csv('DataHOG_PCA20.csv', header=None).to_numpy()
LBPdata = pd.read_csv('DataLBP_PCA20.csv', header=None).to_numpy()
RGBdata = pd.read_csv('DataRGB_PCA20.csv', header=None).to_numpy()
Label = pd.read_csv('Labels.csv', header=None).to_numpy().ravel()
Labels= Label-1

# Question 1 b
kmeans1 = KMeans(n_clusters=3)
kmeans1.fit(HOGdata)

kmeans2 = KMeans(n_clusters=3)
kmeans2.fit(LBPdata)

kmeans3 = KMeans(n_clusters=3)
kmeans3.fit(RGBdata)


labels1 = kmeans1.labels_
labels2 = kmeans2.labels_
labels3 = kmeans3.labels_

clustering1 = AgglomerativeClustering(n_clusters=3)
clustering1.fit(HOGdata)


clustering2 = AgglomerativeClustering(n_clusters=3)
clustering2.fit(LBPdata)

clustering3 = AgglomerativeClustering(n_clusters=3)
clustering3.fit(RGBdata)


labels11=clustering1.labels_ 
labels12=clustering2.labels_
labels13=clustering3.labels_


def confusion_matrix(t, a):
    ut=np.unique(t)
    ul=len(ut)
    cm=np.zeros((ul,ul))
    for i in range(ul):
        for j in range(ul):
            cm[i,j]=np.sum((t==ut[i]) & (a==ut[j]))
    return cm,ut        

def label_matcher(t, a):
    from scipy.optimize import linear_sum_assignment
    cm,ut = confusion_matrix(t, a)
    ri,ci = linear_sum_assignment(-cm)
    na = np.zeros(len(a))
    for i in range(len(ut)):
        na[a== ut[ci[i]]] = ut[ri[i]]
    conf,_ = confusion_matrix(t, na) 
    return np.trace(conf)/np.sum(conf)  

a=labels1
t=Labels
confusion_matrix(t,a)
print('label_matcher1_Kmeans(HOG)=',label_matcher(t,a))

a=labels2
t=Labels
confusion_matrix(t,a) 
print('label_matcher2_Kmeans(LBP)=',label_matcher(t,a))

a=labels3
t=Labels
confusion_matrix(t,a)
print('label_matcher3_Kmeans(RGB)=',label_matcher(t,a))

a=labels11
t=Labels
confusion_matrix(t,a) 
print('label_matcher11_Cluster(HOG)=',label_matcher(t,a))

a=labels12
t=Labels
confusion_matrix(t,a) 
print('label_matcher12_Cluster(LBP)=',label_matcher(t,a))

a=labels13
t=Labels
confusion_matrix(t,a) 
print('label_matcher13_Cluster(RGB)=',label_matcher(t,a))


# Question 1c
# The most successful data representation from label matcher in 1.b is the HOGdata
from sklearn.manifold import TSNE
X = HOGdata
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(HOGdata)
print(X_embedded.shape)


fig, (ax1, ax2)= plt.subplots(1,2)
ax1.scatter(X_embedded[:,0], X_embedded[:,1], c=Labels)
ax1.set_title("True_label")
ax2.scatter(X_embedded[:,0], X_embedded[:,1], c=labels11)
ax2.set_title("HOG_Cluster_label")
plt.show()

#Question 1d


X = [HOGdata]
pca = PCA(n_components=2)
FittedX = pca.fit_transform(HOGdata)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

plt.figure()
plt.scatter(FittedX[:,0], FittedX[:,1], 'k.')
plt.show()

# Question 2
# Demonstrate the work of Sequential Forward Selection (SFS) on the following example. Consider 6 
# features, numbered from 1 to 6. The function that evaluates a given subset below

def subset_quality(feature_set):
    feature_set = np.array(feature_set)
    u = np.sum(np.sin(feature_set)+ np.cos(feature_set))
    return u

features1 = [1,2,3,4,5,6]

for i in features1:
    feature_set=i
    a=subset_quality(feature_set)
    print(i,'  ', a)
    

# feature 1 has the best Value   1    1.381773290676036
    
features2 =[[1,2], [1,3], [1,4], [1,5], [1,6]] 
for j in features2:
    feature_set=j
    b=subset_quality(feature_set)
    print(j,'  ', b)

# The combination of features 1 and 6 produce the best values
# [1, 6]    2.062528079127476

features3 = [[1,6,2], [1,6,3], [1,6,4], [1,6,5]]  
for k in features3:
    feature_set=k
    c=subset_quality(feature_set)
    print(k,'  ', c)

# the best feature subset is [1,6,2] because it return the highest  value
# [1, 6, 2]    2.5556786694060154


