# %% [markdown]
# # Lab 5

# %% [markdown]
# ## Classifier Ensembles

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import tools_4211 as ts

# %% [markdown]
# Task 1:

# %%
Lab5Data = pd.read_csv('Data2D_Lab5.csv', header=None).to_numpy()

Data = Lab5Data[:, :-1]
Labels = Lab5Data[:, -1].ravel()

print('Prior Probability of Class 1 ', np.mean(Labels == 1))
print('Prior Probability of Class 2 ', np.mean(Labels == 2))
print('Prior Probability of Class 3 ', np.mean(Labels == 3))

# %% [markdown]
# Task 2:
# 
# Initialise all models.

# %%

LDA = LinearDiscriminantAnalysis()
DT = DecisionTreeClassifier()
ONN = KNeighborsClassifier(n_neighbors = 1)
SVM = SVC(gamma=0.1, kernel="rbf")
Bag = BaggingClassifier()
RF = RandomForestClassifier()

Classifiers = [LDA, DT, ONN, SVM, Bag, RF]

# %% [markdown]
# Initialise Hold-out.

# %%
TrD, TsD, TrL, TsL = train_test_split(Data, Labels, train_size = 0.5)

# %% [markdown]
# Train and test all models.

# %%
for cla in Classifiers:
    cla.fit(TrD, TrL)
    al = cla.predict(TsD)
    error = np.mean(al != TsL)
    print('Error rate for ', cla.__class__.__name__, ':', error)

# %% [markdown]
# Notice: Ensembles worked better than individual classifiers on the same train/test splits.

# %%
colours= np.array([[1,0,0],[0,1,0],[0,0,1]])
ts.plot_2D_data(Data, Labels, colours)
plt.show()

# %%
for cla in Classifiers:
    name = cla.__class__.__name__
    fig = plt.figure()
    ts.plot_regions(Data, cla, colours)
    plt.title(str(name))
    plt.show()


