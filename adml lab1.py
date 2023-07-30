import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

a=np.random.normal(loc=[0,0], scale=1.0, size=[3000,2])
b=np.random.normal(loc=[0,0], scale=3.0, size=[3000,2])
c=np.random.normal(loc=[0,0], scale=6.0, size=[3000,2])
d=np.random.normal(loc=[0,0], scale=10.0, size=[3000,2])


'''xx=[d, c, b, a]
colour= np.array(['k', 'r', 'b', 'g'])
for j, i in enumerate(xx):
    plt.scatter(i[:,0], i[:,1], c=colour[j])   

plt.grid()
plt.show()'''


rv = multivariate_normal.pdf(mean=[0.5, -0.2], cov=np.array([2.0, 0.3], [0.3, 0.5]))
print(rv)
