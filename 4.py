import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

study_time = np.array([1,2,3,6,8])
pass_flag = np.array([0,0,0,1,1])

test_study_time = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])

# x_{0} = 1, x_{1} = study_time / y = pass_flag: 0=Fail, 1=Pass

def sigmoid(x): # 시그모이드 함수
    z = 1/(1+np.exp(-x))
    return z

def dev_sigmoid(x): # 시그모이드 함수 미분
    z = sigmoid(x)*(1-sigmoid(x))
    return z

def pred(x,w): # pred 즉, numpy 값들과 w 가중치를 통해서 z numpy array를 만들어 준다.
    #print('x and w:', x, w)
    z = w[0] + w[1]*x
    #z = sigmoid(w[0] + w[1]*x)
    return z

def cost(x,y,w): # cost(x,y,w) 즉 x 값, y 값, 가중치 w 값까지 주게되면
    z = np.mean(-y*np.log(sigmoid(pred(x,w)))-(1-y)*np.log(1-sigmoid(pred(x,w))))
    return z

def dev_cost(x,y,w):
    z0 = np.mean(sigmoid(pred(x,w))-y) # x_{0} = 1
    z1 = np.mean(x*(sigmoid(pred(x,w))-y))
    return np.array([z0,z1])


w=np.zeros(2)
epsilon = 1e-3
Y_pred = np.zeros(13)
Lr = 0.05
prev_cost = 0
while (np.abs(cost(study_time,pass_flag,w)-prev_cost)/cost(study_time,pass_flag,w)) > epsilon: 
    prev_cost = cost(study_time,pass_flag,w)
    w=w-Lr*dev_cost(study_time,pass_flag,w)
    #print(w)
Y_pred=sigmoid(pred(test_study_time,w))
Ans=np.copy(Y_pred)
Ans[Ans>=0.5] =1
Ans[Ans<0.5]=0
print(Ans)