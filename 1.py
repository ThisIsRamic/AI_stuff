import numpy as np
X =np.array([1,2,3])
Y=np.array([4,7,10])

w=0
b=0
epsilon = 1e-6
Y_pred = np.zeros(3)
Lr = 0.001

while np.mean(abs(Y-Y_pred))>epsilon:
    w=w-Lr*np.sum(2*np.power(X,2)*w+2*b*X-2*X*Y)/len(X)
    b=b-Lr*np.sum(2*b+2*w*X-2*Y)/len(X)
    Y_pred=w*X+b
    print(np.mean(abs(w*X+b-Y)))

import matplotlib.pyplot as plt
plt.plot(X,Y,c='r')
plt.plot(X,Y_pred,c='b')
plt.show()


