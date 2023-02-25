import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

np.random.seed(1)

study_time = np.array([1,2,3,6,8]).reshape(-1,1)
pass_flag = np.array([0,0,0,1,1])



def sigmoid(x): # 시그모이드 함수
    z = 1/(1+np.exp(-x))
    return z
clf = LogisticRegression(random_state=0).fit(study_time, pass_flag)
test_study_time = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]).reshape(-1,1)

pred_flag=clf.predict(test_study_time)
pred_prob = clf.predict_proba(test_study_time)

print(pred_flag)