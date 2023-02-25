import pandas as pd

boston = pd.read_csv("Boston_house.csv")
boston.head()

from sklearn.linear_model import LinearRegression
import numpy as np
X = boston["CRIM"]
Y = boston["Target"]
w=0
b=0
epsilon = 1e-4
Y_pred = np.zeros(506)
Lr = 0.0001

while np.mean(abs(Y-Y_pred))>epsilon:
    w=w-Lr*np.sum(2*np.power(X,2)*w+2*b*X-2*X*Y)/len(X)
    b=b-Lr*np.sum(2*b+2*w*X-2*Y)/len(X)
    Y_pred=w*X+b
    print(np.mean(abs(w*X+b-Y)))

import matplotlib.pyplot as plt
plt.plot(X,Y,c='r')
plt.plot(X,Y_pred,c='b')
plt.show()