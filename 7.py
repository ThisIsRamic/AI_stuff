import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
a=np.arange(-3,3,0.01)
def Square(x:ndarray) -> ndarray:
    return np.power(x,2)
def leaky_relu(x:ndarray)->ndarray:
    return np.maximum(0.2*x,x)
plt.plot(a,leaky_relu(a))
plt.show()