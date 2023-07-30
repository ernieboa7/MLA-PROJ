import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



def gen_lab_data(N, P1):
    N1=np.round(N*P1).astype(int)
    N2=np.round(N*(1-P1)).astype(int)
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
'''# Define the class-conditional pdfs
def p1(x,y):
    if (-2 < x < 3 and 1 < y < 5):
        return 1/30
    else:
        return 0

def p2(x,y):
    if (1 < x < 5 and 0 < y < 5):
        return 1/20
    else:
        return 0

def p3(x,y):
    mu = [2, 4]
    cov = [[2, -2], [-2, 9]]
    return multivariate_normal.pdf([x,y], mean=mu, cov=cov)

# Define the prior probabilities
P1 = 0.3
P2 = 0.2
P3 = 0.5

# Define the decision boundaries
x = np.linspace(-3, 6, 100)
y = np.linspace(-1, 8, 100)
X, Y = np.meshgrid(x, y)

Z1 = p1(X,Y)*P1 - p2(X,Y)*P2
Z2 = p1(X,Y)*P1 - p3(X,Y)*P3
Z3 = p2(X,Y)*P2 - p3(X,Y)*P3

# Plot the decision boundaries
plt.contour(X, Y, Z1, levels=[0], colors='red')
plt.contour(X, Y, Z2, levels=[0], colors='blue')
plt.contour(X, Y, Z3, levels=[0], colors='green')

# Label the plot
plt.title('Bayes Classifier Decision Boundaries')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-3, 6])
plt.ylim([-1, 8])
plt.show()'''