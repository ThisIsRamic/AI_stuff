from sklearn.linear_model import LinearRegression
import numpy as np
X =np.array([1,2,3])
Y=np.array([4,7,10])
reg = LinearRegression().fit(X,Y)
print(reg.coef_)