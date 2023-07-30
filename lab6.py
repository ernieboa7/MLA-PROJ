# %% [markdown]
# # Lab 6 Solutions

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# %% [markdown]
# ## Question 1
# 
# - Load the datafiles

# %%
CData1 = pd.read_csv('ClusteringData1.csv', header = None).to_numpy()
CData2 = pd.read_csv('ClusteringData2.csv', header = None).to_numpy()
CData3 = pd.read_csv('ClusteringData3.csv', header = None).to_numpy()

Data = [CData1, CData2, CData3]
ClusterNo = [2, 2, 3]

# %% [markdown]
# - Plot KMeans and Single Linkage for each Data-Set

# %%
colours = np.array([[1,0,0],[0,1,0],[0,0,1]])

for i in range(len(Data)):
    Set = Data[i]
    Kmeans = KMeans(n_clusters=ClusterNo[i])
    Kmeans.fit(Set)
    SingleLinkage = AgglomerativeClustering(linkage='single', n_clusters=ClusterNo[i])
    SingleLinkage.fit(Set)

    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("KMeans")
    ax1.axis('equal')
    ax2.set_title("Single Linkage")
    ax2.axis('equal')

    uq = np.unique(Kmeans.labels_)
    for j in uq:
        ax1.plot(Set[Kmeans.labels_ == j, 0], Set[Kmeans.labels_ == j, 1], '.', c=colours[j])
        ax1.plot(Kmeans.cluster_centers_[j, 0], Kmeans.cluster_centers_[j, 1], 'x', c=colours[j])
        ax2.plot(Set[SingleLinkage.labels_ == j, 0], Set[SingleLinkage.labels_ == j, 1], '.', c=colours[j])



# %% [markdown]
# ## Question 2
# 
# - Perform SFS on 6 features to find the best trio.
# - I<sub>i</sub> = 1 when in the set, and 0 otherwise.
# - 𝑓(𝑆) = 5𝐼! 	 + 	4𝐼" 	 − 	2𝐼$ 	 − 	8𝐼% 	 + 	2	𝐼& 	 − 	5	𝐼#

# %%
Selected = []
Available = [1, 2, 3, 4, 5, 6]

while len(Selected) < 3:
    print('Selected ', Selected)
    print('Available ', Available)

    Best = -1
    FBest = -1
    for i in Available:
        NewSet = list([i])
        NewSet.extend(Selected)
        ISet = [0, 0, 0, 0, 0, 0]
        for j in NewSet:
            ISet[j-1] = 1

        F = (5*ISet[0])+(4*ISet[1])-(2*ISet[2])-(8*ISet[3])+(2*ISet[4])-(5*ISet[5])
        print('  Testing ', NewSet, ', F Value ', F)
        if F > FBest:
            Best = i
            FBest = F
    
    if Best != -1:
        Selected.append(Best)
        Available.remove(Best)
        
print('\nFinal Selection ', Selected)
print('F Value ', FBest)


