import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
irisfile= pd.read_csv('iris.csv.txt', header=None)
irisarray = irisfile.to_numpy()

print(irisfile)
print(irisarray[:, :-1])
Data_split= irisarray[:, :-1]

print(Data_split)
Labels_split= irisarray[:, -1]
print(Labels_split)

for i in range(4):
    for j in range(4):
        plt.figure()
        plt.scatter(Data_split[:, i], Data_split[:, j], c=Labels_split)
        
        
plt.show()