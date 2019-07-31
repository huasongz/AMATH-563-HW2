#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:33:39 2019

@author: huasongzhang
"""

import numpy as np
import h5py 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

f = h5py.File('BZ.mat','r') 
data = f.get('BZ_tensor') 
data = np.array(data)
[a,b,c] = data.shape
data2 = np.zeros((a,b*c))
x = []

for i in range(a):
    o = data[i,:,:]
    data2[i,:] = np.reshape(data[i,:,:], (1, b*c))
    data2[i,:] = data2[i,:]/max(data2[i,:])
    x.append(np.argmax(data2[i,:]))
 
    
x = np.asarray(x)
l = len(x)
x1s = x[0:l-1]



# build library
def LAB(x1):
    M = np.array([x1,x1,x1**2,x1**3,x1**4,x1**5,np.sin(x1),np.cos(x1),
                  np.log(x1),np.sqrt(x1)])
    if isinstance(x1, int) or isinstance(x1, float):
        M[0] = 1
    else:
        M[0] = np.ones(len(x1))
    #M = np.squeeze(M).T
    return M

A = LAB(x1s).T




# pinv
xi1 = np.dot(np.linalg.pinv(A),x1dot)
# LASSO
clf_l = linear_model.Lasso(alpha=0.01)
clf_l.fit(A, x1dot)
xi2 = clf_l.coef_
# Ridge
clf_r = Ridge(alpha=0.05)
clf_r.fit(A, x1dot) 
xi3 = clf_r.coef_

f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12, 4))
ax1.bar(range(len(xi1)),xi1)
ax1.set_title('Weights using pinv')
ax2.bar(range(len(xi2)),xi2)
ax2.set_title('Weights using LASSO')
ax3.bar(range(len(xi3)),xi3)
ax3.set_title('Weights using Ridge')
f.suptitle('Weights')
f.savefig('1d.jpg')

