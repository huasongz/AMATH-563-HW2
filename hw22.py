#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:55:39 2019

@author: huasongzhang
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

population = scipy.io.loadmat('population.mat')
x1 = population['hare'].ravel()
x2 = population['lynx'].ravel()
t = population['year'].ravel()
dt = t[1]-t[0]


spl1 = UnivariateSpline(t, x1)
spl2 = UnivariateSpline(t, x2)
t_new = np.arange(1845,1903,0.2)
x1_new = spl1(t_new)
x2_new = spl2(t_new)
dt = t_new[1]-t_new[0]

plt.plot(t_new,x1_new)
plt.plot(t,x1,'ro')

  


  
# compute the differences
x1s = x1_new[0:-1]
x1dot = x1_new[1:]
x2s = x2_new[0:-1]
x2dot = x2_new[1:]




# create library
#A = np.array([np.ones(len(x1s)),x1s,x2s,x1s**2,x1s*x2s,x2s**2,x1s**3,
#              x1s*(x2s**2),(x1s**2)*x2s,x2s**3])
#A = np.array([np.ones(len(x1s)),x1s,x2s,x1s**2,x1s*x2s,x2s**2,x1s**3,
#              x1s*(x2s**2),(x1s**2)*x2s,x2s**3,np.sin(x1s),np.cos(x1s),
#              np.sin(x2s),np.cos(x2s),np.sin(x1s)*np.cos(x2s),
#              np.cos(x1s)*np.sin(x2s)])
#A = np.squeeze(A).T

def LAB(x1,x2):
    M = np.array([x1,x1,x2,x1**2,x1*x2,x2**2,x1**3,x1*(x2**2),(x1**2)*x2,
                  x2**3,np.sin(x1),np.cos(x1),np.sin(x2),np.cos(x2),
                  np.sin(x1)*np.cos(x2),np.cos(x1)*np.sin(x2)])
    if isinstance(x1, int) or isinstance(x1, float):
        M[0] = 1
    else:
        M[0] = np.ones(len(x1))
    M = np.squeeze(M).T
    return M

A = LAB(x1s,x2s)
    





# least square fit
xi11 = np.dot(np.linalg.pinv(A),x1dot)
xi12 = np.dot(np.linalg.pinv(A),x2dot)
l = len(xi11)
f, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 4))
ax1.bar(range(1,l+1),xi11)
ax1.set_title('Weights for hare')
ax2.bar(range(1,l+1),xi12)
ax2.set_title('Weights for lynx')
f.suptitle('Weights by using pinv')
f.savefig('pinv.jpg')

a1 = np.where(abs(xi11)>0.2)
b1 = np.where(abs(xi12)>0.5)
#x1_pre1 = np.sum(np.squeeze(A[:,a1])*xi11[a1],axis = 1)
#x2_pre1 = np.sum(np.squeeze(A[:,b1])*xi12[b1],axis = 1)
#err11 = np.sum((x1_pre1-x1dot)**2)
#err12 = np.sum((x2_pre1-x2dot)**2)

#plt.figure(4)
#plt.plot(x1dot,'r',label="d(hare)")
#plt.plot(x1_pre1,'b',label="d(lynx)")
#plt.legend()
#plt.xlabel('time')
#plt.ylabel('hare population')
#
#plt.figure(5)
#plt.plot(x2dot,'r',label="d(hare)")
#plt.plot(x2_pre1,'b',label="d(lynx)")
#plt.legend()
#plt.xlabel('time')
#plt.ylabel('lynx population')




Aa1_new = A[:,a1]
Aa1_new = np.squeeze(Aa1_new)
Ab1_new = A[:,b1]
Ab1_new = np.squeeze(Ab1_new)
xi11_new = np.dot(np.linalg.pinv(Aa1_new),x1dot)
xi12_new = np.dot(np.linalg.pinv(Ab1_new),x1dot)
x1_pre11 = np.sum(Aa1_new*xi11_new,axis = 1)
x2_pre11 = np.sum(Ab1_new*xi12_new,axis = 1)
err11 = np.sum((x1_pre11-x1dot)**2)
err12 = np.sum((x2_pre11-x2dot)**2)

#plt.figure(4)
#plt.plot(x1dot,'r',label="d(hare)")
#plt.plot(x1_pre11,'b',label="d(lynx)")
#plt.legend()
#plt.xlabel('time')
#plt.ylabel('hare population')
#
#plt.figure(5)
#plt.plot(x2dot,'r',label="d(hare)")
#plt.plot(x2_pre11,'b',label="d(lynx)")
#plt.legend()
#plt.xlabel('time')
#plt.ylabel('lynx population')

def predict(x,y,n,a,b,xi,yi,dt):
    x_pre = np.zeros(n)
    y_pre = np.zeros(n)
    x_pre[0] = x
    y_pre[0] = y
    for i in range(n-1):
#        x = sum(LAB(x,y)*r1*xi*dt)+x
#        y = sum(LAB(x,y)*r2*yi*dt)+y
        x = sum(np.diff(LAB(x,y)[a]*xi)*dt)+x
        y = sum(np.diff(LAB(x,y)[b]*yi)*dt)+y
        x_pre[i+1] = x
        y_pre[i+1] = y
    return (x_pre,y_pre)
    
x1_pre_1,x2_pre_1 = predict(20,32,30,a1,b1,xi11_new,xi12_new,2)

plt.figure(9)
plt.plot(x1_pre_1,'ro')
plt.plot(x1,'b')
plt.figure(10)
plt.plot(x2_pre_1,'ro')
plt.plot(x2,'b')


'''
# LASSO
clf_l1 = linear_model.Lasso(alpha=0.05)
clf_l1.fit(A, x1dot)
xi21 = clf_l1.coef_
clf_l2 = linear_model.Lasso(alpha=0.05)
clf_l2.fit(A, x2dot)
xi22 = clf_l2.coef_
f, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 4))
ax1.bar(range(1,l+1),xi21)
ax1.set_title('Weights for hare')
ax2.bar(range(1,l+1),xi22)
ax2.set_title('Weights for lynx')
f.suptitle('Weights by using LASSO')
f.savefig('LASSO.jpg')

a2 = np.where(abs(xi21)>0.2)
b2 = np.where(abs(xi22)>0.1)
x1_pre2 = np.sum(np.squeeze(A[:,a2])*xi21[a2],axis = 1)
x2_pre2 = np.sum(np.squeeze(A[:,b2])*xi22[b2],axis = 1)
err21 = np.sum((x1_pre2-x1dot)**2)
err22 = np.sum((x2_pre2-x2dot)**2)




# Ridge
clf_r1 = Ridge(alpha=0.05)
clf_r1.fit(A, x1dot) 
xi31 = clf_r1.coef_
clf_r2 = Ridge(alpha=0.05)
clf_r2.fit(A, x2dot) 
xi32 = clf_r2.coef_
f, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 4))
ax1.bar(range(1,l+1),xi31)
ax1.set_title('Weights for hare')
ax2.bar(range(1,l+1),xi32)
ax2.set_title('Weights for lynx')
f.suptitle('Weights by using Ridge')
f.savefig('Ridge.jpg')

a3 = np.where(abs(xi31)>0.2)
b3 = np.where(abs(xi32)>0.1)
x1_pre3 = np.sum(np.squeeze(A[:,a3]*xi31[a3]),axis = 1)
x2_pre3 = np.sum(np.squeeze(A[:,b3]*xi32[b3]),axis = 1)
err31 = np.sum((x1_pre3-x1dot)**2)
err32 = np.sum((x2_pre3-x2dot)**2)

Err = np.array([[err11,err12],[err21,err22],[err31,err32]])
min1 = np.argmin(Err[:,0])
min2 = np.argmin(Err[:,1])






def predict(x,y,n,a,b,xi,yi):
    x_pre = np.zeros(n)
    y_pre = np.zeros(n)
    x_pre[0] = x
    y_pre[0] = y
    for i in range(n-1):
        x = sum(LAB(x,y)[a]*xi[a])
        y = sum(LAB(x,y)[b]*yi[b])
#        x = 3.90063713 + 1.10683591*x - 0.40776777*y
#        y = 1.40150221 + 0.85082205*y
        x_pre[i+1] = x
        y_pre[i+1] = y
    return (x_pre,y_pre)
    
x1_pre_1,x2_pre_1 = predict(20,32,30,a1,b1,xi11,xi12)
x1_pre_2,x2_pre_2 = predict(20,32,30,a2,b2,xi21,xi22)
x1_pre_3,x2_pre_3 = predict(20,32,30,a3,b3,xi31,xi32)
plt.figure(9)
plt.plot(x1_pre_1,'ro')
plt.plot(x1_pre_2,'go')
plt.plot(x1_pre_3,'yo')
plt.plot(x1,'b')
plt.figure(10)
plt.plot(x2_pre_1,'ro')
plt.plot(x2_pre_2,'go')
plt.plot(x2_pre_3,'yo')
plt.plot(x2,'b')
'''