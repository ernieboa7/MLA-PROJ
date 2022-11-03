import numpy as np
import matplotlib.pyplot as plt
import tools_4211 as ts
from sklearn.model_selection import train_test_split


dataz, labelsz = ts.generate_two_spirals(N1=250, N2=250, noise=0.05, revolutions=3, randomise_step=True)
plt.figure()
plt.scatter(dataz[:,0], dataz[:, 1], c=labelsz, s=8)
plt.grid('On')
plt.show()

data, labels = ts.generate_two_spirals(N1=15000, N2=15000, noise=0.05, revolutions=3, randomise_step=True)
plt.figure()
plt.scatter(data[:,0], data[:, 1], c=labels, s=8)
plt.grid('On')
plt.show()

trd, tsd, trl, tsl = train_test_split(data,labels, test_size=0.5)

#testing_errors, assigned_labels, cla = ts.train_test_tree(data, labels,
                                           #data, labels)