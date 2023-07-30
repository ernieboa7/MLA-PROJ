import numpy as np
import matplotlib.pyplot as plt



class1 = np.random.multivariate_normal([0, 0], [[2, -1], [-1, 11]], size=5000)
class2a = np.random.multivariate_normal([1, -1], [[4, 2], [2, 3]], size=2500)
class2b = np.random.multivariate_normal([-2, 1], [[2, -1], [-1, 11]], size=2500)
data2 = np.concatenate((class2a, class2b))



labels = np.ravel(np.concatenate((np.ones([5000,1]), np.ones([5000,1])*2), axis=0))
data1 = np.concatenate((class1, data2))



def gen_lab_data(N, P1):
    N1=np.round(N*P1).astype(int)
    N2=np.round(N*(1-P1)/2).astype(int)
    S1= np.array([[2, -1], [-1, 11]])
    S2= np.array([[4, 2], [2, 3]])
    M1=[0,0]
    M2=[1,-1]
    M3=[-2,1]
    C1 = np.random.multivariate_normal(M1, S1, size=N1)
    C2a = np.random.multivariate_normal(M2, S2, size=N2)
    C2b = np.random.multivariate_normal(M3, S1, size=N2)
    D= np.vstack((C1,C2a,C2b))
    L1= np.zeros([N1,1]).ravel()
    L2= np.ones([2*N2,1]).ravel()
    L=np.hstack((L1,L2))

    return(D, L)             
                 
fulldata, fulllabels = gen_lab_data(10000, 0.5)
testdata1, testlabels1 = gen_lab_data(10000, 0.8)
testdata2, testlabels2 = gen_lab_data(10000, 0.998)


fulldata, fulllabels = gen_lab_data(50000, 0.5)
testdata1, testlabels1 = gen_lab_data(50000, 0.8)
testdata2, testlabels2 = gen_lab_data(50000, 0.998)


print(testdata1.shape, testdata2.shape, '\n', testdata2.shape, testlabels2.shape)
print(np.mean(testlabels1==0), np.mean(testlabels2))

'''colours = np.array(['k', 'r'])

for count, label in enumerate(np.unique(labels1)):
   plt.scatter(fulldata[fulllabels==label,0], fulldata[fulllabels==label,1], s=1, c=colours[count])
plt.grid('on')
plt.ylabel('y')
plt.xlabel('x')
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.show()'''




