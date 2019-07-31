#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:50:36 2019

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
X = []
Y = []
for i in range(a):
    o = data[i,:,:]
    data2[i,:] = np.reshape(data[i,:,:], (1, b*c))
    data2[i,:] = data2[i,:]/max(data2[i,:])
    x.append(np.argmax(data2[i,:]))
    
    o = o/(o.max())
    [xx,yy] = np.unravel_index(np.argmax(o, axis=None), o.shape)
    X.append(xx)
    Y.append(yy)    
    
x = np.asarray(x)
X = np.asarray(X)
Y = np.asarray(Y)
l = len(X)
x1s = X[0:l-1]
x1dot = X[1:l]
x2s = Y[0:l-1]
x2dot = Y[1:l]




# build library2
def LAB(x1,x2):
    M = np.array([x1,x1,x2,x1**2,x1*x2,x2**2,x1**3,x1*(x2**2),(x1**2)*x2,
                  x2**3,np.sin(x1),np.cos(x1),np.sin(x2),np.cos(x2),
                  np.sin(x1)*np.cos(x2),np.cos(x1)*np.sin(x2)])
    if isinstance(x1, int) or isinstance(x1, float) or isinstance(x1,np.int64) :
        M[0] = 1
    else:
        M[0] = np.ones(len(x1))
    M = np.squeeze(M).T
    return M

A = LAB(x1s,x2s)


# pinv
xi11 = np.dot(np.linalg.pinv(A),x1dot)
xi12 = np.dot(np.linalg.pinv(A),x2dot)
# LASSO
clf_l1 = linear_model.Lasso(alpha=0.01)
clf_l1.fit(A, x1dot)
xi21 = clf_l1.coef_
clf_l2 = linear_model.Lasso(alpha=0.01)
clf_l2.fit(A, x2dot)
xi22 = clf_l2.coef_
# Ridge
clf_r1 = Ridge(alpha=0.05)
clf_r1.fit(A, x1dot) 
xi31 = clf_r1.coef_
clf_r2 = Ridge(alpha=0.05)
clf_r2.fit(A, x2dot) 
xi32 = clf_r2.coef_

f, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 4))
ax1.bar(range(len(xi11)),xi11)
ax1.set_title('weights on x')
ax2.bar(range(len(xi12)),xi12)
ax2.set_title('weights on y')
f.suptitle('Weights using pinv')
f.savefig('2d_pinv.jpg')
a1 = np.where(abs(xi11)>1)
b1 = np.where(abs(xi12)>1)
r = np.zeros(16)
r1 = r
r2 = r
r1[a1] = 1
r2[b1] = 1
l1 = len(a1[0])
l2 = len(b1[0])
B1 = A*r1
B2 = A*r2

for i in range(5):
    xi11_new = np.dot(np.linalg.pinv(B1),x1dot)
    xi12_new = np.dot(np.linalg.pinv(B2),x2dot)
    a1 = np.where(abs(xi11_new)>1)
    print(a1)
    b1 = np.where(abs(xi12_new)>1)
    print(b1)
    if len(a1[0])  == l1 and len(b1[0]) == l2:
        break
    else:
        r = np.zeros(16)
        r1 = r
        r2 = r
        r1[a1] = 1
        r2[b1] = 1
        B1 = A*r1
        B2 = A*r2
        l1 = len(a1[0])
        l2 = len(b1[0])
        
x1_pre1 = np.sum(B1*xi11_new,axis = 1)
x2_pre1 = np.sum(B2*xi12_new,axis = 1)
err11 = np.sum((x1_pre1-x1dot)**2)
err12 = np.sum((x2_pre1-x2dot)**2)


f, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 4))
ax1.bar(range(len(xi21)),xi21)
ax1.set_title('weights on x')
ax2.bar(range(len(xi22)),xi22)
ax2.set_title('weights on y')
f.suptitle('Weights using LASSO')
f.savefig('2d_LASSO.jpg')
a2 = np.where(abs(xi21)>1)
b2 = np.where(abs(xi22)>0.5)
r = np.zeros(16)
r1 = r
r2 = r
r1[a2] = 1
r2[b2] = 1
l1 = len(a2[0])
l2 = len(b2[0])
B1 = A*r1
B2 = A*r2

for i in range(5):
    clf_l1 = linear_model.Lasso(alpha=0.01)
    clf_l1.fit(B1, x1dot)
    xi21_new = clf_l1.coef_
    clf_l2 = linear_model.Lasso(alpha=0.01)
    clf_l2.fit(B2, x2dot)
    xi22_new = clf_l2.coef_
    a2 = np.where(abs(xi21_new)>1)
    print(a2)
    b2 = np.where(abs(xi22_new)>0.5)
    print(b2)
    if len(a2[0])  == l1 and len(b2[0]) == l2:
        break
    else:
        r = np.zeros(16)
        r1 = r
        r2 = r
        r1[a2] = 1
        r2[b2] = 1
        B1 = A*r1
        B2 = A*r2
        l1 = len(a2[0])
        l2 = len(b2[0])

x1_pre2 = np.sum(B1*xi21_new,axis = 1)
x2_pre2 = np.sum(B2*xi22_new,axis = 1)
err21 = np.sum((x1_pre2-x1dot)**2)
err22 = np.sum((x2_pre2-x2dot)**2)



f, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 4))
ax1.bar(range(len(xi31)),xi31)
ax1.set_title('weights on x')
ax2.bar(range(len(xi32)),xi32)
ax2.set_title('weights on y')
f.suptitle('Weights using Ridge')
f.savefig('2d_Ridge.jpg')

a3 = np.where(abs(xi31)>1)
b3 = np.where(abs(xi32)>0.5)
r = np.zeros(16)
r1 = r
r2 = r
r1[a3] = 1
r2[b3] = 1
l1 = len(a3[0])
l2 = len(b3[0])
B1 = A*r1
B2 = A*r2

for i in range(5):
    clf_r1 = Ridge(alpha=0.05)
    clf_r1.fit(A, x1dot) 
    xi31_new = clf_r1.coef_
    clf_r2 = Ridge(alpha=0.05)
    clf_r2.fit(A, x2dot) 
    xi32_new = clf_r2.coef_
    a3 = np.where(abs(xi31_new)>1)
    print(a3)
    b3 = np.where(abs(xi32_new)>0.5)
    print(b3)
    if len(a3[0])  == l1 and len(b3[0]) == l2:
        break
    else:
        r = np.zeros(16)
        r1 = r
        r2 = r
        r1[a3] = 1
        r2[b3] = 1
        B1 = A*r1
        B2 = A*r2
        l1 = len(a3[0])
        l2 = len(b3[0])

x1_pre3 = np.sum(B1*xi31_new,axis = 1)
x2_pre3 = np.sum(B2*xi32_new,axis = 1)
err31 = np.sum((x1_pre3-x1dot)**2)
err32 = np.sum((x2_pre3-x2dot)**2)


def predict(x,y,n,a,b,xi,yi,dt):
    x_pre = np.zeros(n)
    y_pre = np.zeros(n)
    x_pre[0] = x
    y_pre[0] = y
    r = np.zeros(16)
    r1 = r
    r2 = r
    r1[a] = 1
    r2[b] = 1
    for i in range(n-1):
        x = sum(LAB(x,y)*r1*xi)
        y = sum(LAB(x,y)*r2*yi)
#        x = sum(LAB(x,y)*r1*xi*dt)+x
#        y = sum(LAB(x,y)*r2*yi*dt)+y
        x_pre[i+1] = x
        y_pre[i+1] = y
    return (x_pre,y_pre)
    
x1_pre_1,x2_pre_1 = predict(X[0],Y[0],len(X),a1,b1,xi11_new,xi12_new,2)
x1_pre_2,x2_pre_2 = predict(X[0],Y[0],len(X),a2,b2,xi21_new,xi22_new,2)
x1_pre_3,x2_pre_3 = predict(X[0],Y[0],len(X),a3,b3,xi31_new,xi32_new,2)