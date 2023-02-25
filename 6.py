import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

# train data (XOR Problem)
x1 = np.array([0,1,0,1])
x2 = np.array([0,0,1,1])
y = np.array([0,1,1,0])

# Intialization

# input - hidden layer
w1_x1 = np.random.randn(2,1)   
w1_x2 = np.random.randn(2,1)   

# hidden - output layer
w2_h1 = np.random.randn(1)
w2_h2 = np.random.randn(1)

# epoch
ep = 20000
# learning rate
lr = 1
mse = []
E=0

# Neural Networks 2-2-1
for i in range(ep):
    E=0
    for j in range(4):
        net_h1 = np.sum(x1[j]*w1_x1[0]+x2[j]*w1_x2[0])
        net_h2 = np.sum(x1[j]*w1_x1[1]+x2[j]*w1_x2[1])
    h1 = sigmoid(net_h1)
    h2 = sigmoid(net_h2)
    net_o = h1*w2_h1 + h2*w2_h2
    On=sigmoid(net_o)
    for j in range(4):
        E= E+np.mean(abs(y[j] -On)**2)
    print(E)
    for j in range(4):
        w1_x1[0] = w1_x1[0] - lr*((y[j] - On)*On*(1-On)*w2_h1*h1*(1-h1)*x1[j])
        w1_x1[1] = w1_x1[1] - lr*((y[j] - On)*On*(1-On)*w2_h2*h2*(1-h2)*x1[j])
        w1_x2[0] = w1_x2[0] - lr*((y[j] - On)*On*(1-On)*w2_h1*h1*(1-h1)*x2[j])
        w1_x2[1] = w1_x2[1] - lr*((y[j] - On)*On*(1-On)*w2_h2*h2*(1-h2)*x2[j])
    for j in range(4):
        w2_h1 = w2_h1 - lr*((y[j] - On)*On*(1-On)*h1)
        w2_h2 = w2_h2 - lr*((y[j] - On)*On*(1-On)*h2)
    