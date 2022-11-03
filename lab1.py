
import numpy as np
import matplotlib.pyplot as plt
x=np.arange(30, 0, -1)

y=np.random.random(30)


plt.figure()
plt.plot(y,x,'c.')
#plt.axis('Equal')
#plt.grid('On')
plt.show()

# replace all element in y smaller than 0.4 with 1
y[y<0.4] =1

plt.figure()
plt.plot(y,x,'m.')
#plt.axis('Equal')
#plt.grid('On')
plt.show()


reshaped_x=x.reshape(5, -1)
print(reshaped_x)

reshaped_y=y.reshape(5, -1)
print(reshaped_y)


plt.imshow(reshaped_y)
plt.show()
plt.imshow(reshaped_x)
plt.show()


#plt.figure()
#for i in y:
    #for j in x:
        #plt.plot(y, x, 'k.')
        #plt.show()

        