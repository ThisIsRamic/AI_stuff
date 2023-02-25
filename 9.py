from typing import Callable
from typing import List
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

def deriv(func:Callable[[ndarray],ndarray],input_:ndarray,delta:float = 0.001)->ndarray:
    return (func(input_+delta)-func(input_ - delta))/(2*delta)

Array_func = Callable[[ndarray],ndarray]
Chain = List[Array_func]

def chain_length_2(chain:Chain, a:ndarray)->ndarray:
    assert len(chain)==2,"wtf"
    f1=chain[1]
    f2=chain[0]
    return f2(f1(a))

def sigmoid(x:ndarray)->ndarray: # 시그모이드 함수
    z = 1/(1+np.exp(-x))
    return z

def Square(x:ndarray) -> ndarray:
    return np.power(x,2)

def leaky_relu(x:ndarray)->ndarray:
    return np.maximum(0.2*x,x)

def chain_deriv_2(chain:Chain, input_range:ndarray)-> ndarray:
    assert len(chain)==2,"lol"
    assert input_range.ndim == 1,"lmao"
    f1 = chain[1]
    f2 = chain[0]

    df1dx=deriv(f1,input_range)
    df2du = deriv(f2,f1(input_range))
    return df2du*df1dx

def chain_deriv_3(chain:Chain, input_range:ndarray)-> ndarray:
    assert len(chain)==3,"lol"
    f1 = chain[0]
    f2 = chain[1]
    f3=chain[2]

    df1dx=deriv(f1,input_range)
    df2du = deriv(f2,f1(input_range))
    df3dl=deriv(f3,f2(input_range))

    return df3dl*df2du*df1dx

def chain_length_3(chain:Chain, a:ndarray)->ndarray:
    assert len(chain)==3,"wtf"
    f1=chain[0]
    f2=chain[1]
    f3=chain[2]
    return f3(f2(f1(a)))

def multi_inputs_add(x:ndarray, y:ndarray,sigma:Array_func)->float:
    assert x.shape == y.shape

    z=x+y
    return sigma(z)

a=np.arange(-3,3,0.01)
chain_1 = [leaky_relu,sigmoid,Square]
plt.plot(a,chain_length_3(chain_1,a))
plt.plot(a,chain_deriv_3(chain_1,a))
plt.show()