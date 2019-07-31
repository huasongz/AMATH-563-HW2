#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:46:43 2019

@author: huasongzhang
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy.linalg import hankel
#from collections import Counter

population = scipy.io.loadmat('population.mat')
x1 = population['hare'].ravel()
x2 = population['lynx'].ravel()
t = population['year'].ravel()



# interpolate data
spl1 = UnivariateSpline(t, x1)
spl2 = UnivariateSpline(t, x2)
t_new = np.arange(1845,1903.1,0.2)
x1_new = spl1(t_new)
x2_new = spl2(t_new)
dt = t_new[1]-t_new[0]

# compute the derivative
n = len(t_new);
x1dot = np.zeros(n-2)
x2dot = np.zeros(n-2)
for j in range(1,n-1):
    x1dot[j-1] = (x1_new[j+1]-x1_new[j-1])/(2*dt)
    x2dot[j-1] = (x2_new[j+1]-x2_new[j-1])/(2*dt)

# make the measurement the same size of derivative
x1s = x1_new[1:n-1]
x2s = x2_new[1:n-1]


# build library
def LAB(x1,x2):
    M = np.array([x1,x1,x2,x1**2,x1*x2,x2**2,x1**3,x1*(x2**2),(x1**2)*x2,
              x2**3,np.sin(x1),np.cos(x1),np.sin(x2),np.cos(x2),
              np.sin(x1)*np.cos(x2),np.cos(x1)*np.sin(x2)])
    if isinstance(x1, int) or isinstance(x1, float):
        M[0] = 1
    else:
        M[0] = np.ones(len(x1))
    return M.T

A = LAB(x1s,x2s)
[g,h] = A.shape




# least square fit
xi11 = np.dot(np.linalg.pinv(A),x1dot)
xi12 = np.dot(np.linalg.pinv(A),x2dot)
f, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 4))
l = len(xi11)
ax1.bar(range(1,l+1),xi11)
ax1.set_title('Weights for hare')
ax2.bar(range(1,l+1),xi12)
ax2.set_title('Weights for lynx')
f.suptitle('Weights by using pinv')
f.savefig('pinv.jpg')

a1 = np.where(abs(xi11)>0.5)
b1 = np.where(abs(xi12)>0.5)
r1 = np.zeros(h)
r2 = np.zeros(h)
r1[a1] = 1
r2[b1] = 1
l1 = len(a1[0])
l2 = len(b1[0])
for i in range(5):
    xi11_new = np.dot(np.linalg.pinv(A*r1),x1dot)
    xi12_new = np.dot(np.linalg.pinv(A*r2),x2dot)
    a11 = np.where(abs(xi11_new)>0.5)
    b11 = np.where(abs(xi12_new)>0.5)
    if len(a11[0]) == l1 and len(b11[0]) == l2:
        break
    else:
        r1 = np.zeros(h)
        r2 = np.zeros(h)
        r1[a11] = 1
        r2[b11] = 1
r11 = r1
r12 = r2
x1_pre11 = np.sum(A*r11*xi11_new,axis = 1)
x2_pre11 = np.sum(A*r12*xi12_new,axis = 1)
err11 = np.sum((x1_pre11-x1dot)**2)
err12 = np.sum((x2_pre11-x2dot)**2)

f, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 4))
ax1.plot(x1dot,'r',label="derivative of hare(prediction)")
ax1.plot(x1_pre11,'b',label="derivative of hare(real)")
ax1.legend()
ax2.plot(x2dot,'r',label="derivative of hare(prediction)")
ax2.plot(x2_pre11,'b',label="derivative of lynx(real)")
ax2.legend()
f.suptitle('Comparison of derivatives using pinv')
f.savefig('derivative_pinv.jpg')







# LASSO
clf_l1 = linear_model.Lasso(alpha=0.01)
clf_l1.fit(A, x1dot)
xi21 = clf_l1.coef_
clf_l2 = linear_model.Lasso(alpha=0.01)
clf_l2.fit(A, x2dot)
xi22 = clf_l2.coef_
f, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 4))
ax1.bar(range(1,l+1),xi21)
ax1.set_title('Weights for hare')
ax2.bar(range(1,l+1),xi22)
ax2.set_title('Weights for lynx')
f.suptitle('Weights by using LASSO')
f.savefig('LASSO.jpg')

a2 = np.where(abs(xi21)>0.5)
b2 = np.where(abs(xi22)>0.2)
r1 = np.zeros(h)
r2 = np.zeros(h)
r1[a2] = 1
r2[b2] = 1
l1 = len(a2[0])
l2 = len(b2[0])
for i in range(5):
    clf_l11 = linear_model.Lasso(alpha=0.01)
    clf_l11.fit(A*r1, x1dot)
    xi21_new = clf_l11.coef_
    clf_l22 = linear_model.Lasso(alpha=0.01)
    clf_l22.fit(A*r2, x2dot)
    xi22_new = clf_l22.coef_
    a22 = np.where(abs(xi21_new)>0.5)
    b22 = np.where(abs(xi22_new)>0.5)
    if len(a22[0]) == l1 and len(b22[0]) == l2:
        break
    else:
        r1 = np.zeros(h)
        r2 = np.zeros(h)
        r1[a22] = 1
        r2[b22] = 1
r21 = r1
r22 = r2
x1_pre2 = np.sum(A*r21*xi21_new,axis = 1)
x2_pre2 = np.sum(A*r22*xi22_new,axis = 1)
err21 = np.sum((x1_pre2-x1dot)**2)
err22 = np.sum((x2_pre2-x2dot)**2)

f, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 4))
ax1.plot(x1dot,'r',label="derivative of hare(prediction)")
ax1.plot(x1_pre2,'b',label="derivative of hare(real)")
ax1.legend()
ax2.plot(x2dot,'r',label="derivative of hare(prediction)")
ax2.plot(x2_pre2,'b',label="derivative of lynx(real)")
ax2.legend()
f.suptitle('Comparison of derivatives using LASSO')
f.savefig('derivative_LASSO.jpg')







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

a3 = np.where(abs(xi31)>0.5)
b3 = np.where(abs(xi32)>0.2)
r1 = np.zeros(h)
r2 = np.zeros(h)
r1[a3] = 1
r2[b3] = 1
l1 = len(a3[0])
l2 = len(b3[0])
for i in range(5):
    clf_r11 = Ridge(alpha=0.01)
    clf_r11.fit(A*r1, x1dot)
    xi31_new = clf_r11.coef_
    clf_r22 = Ridge(alpha=0.01)
    clf_r22.fit(A*r2, x2dot)
    xi32_new = clf_r22.coef_
    a33 = np.where(abs(xi31_new)>0.5)
    b33 = np.where(abs(xi32_new)>0.5)
    if len(a33[0]) == l1 and len(b33[0]) == l2:
        break
    else:
        r1 = np.zeros(h)
        r2 = np.zeros(h)
        r1[a33] = 1
        r2[b33] = 1
r31 = r1
r32 = r2
x1_pre3 = np.sum(A*r31*xi31_new,axis = 1)
x2_pre3 = np.sum(A*r31*xi31_new,axis = 1)
err31 = np.sum((x1_pre3-x1dot)**2)
err32 = np.sum((x2_pre3-x2dot)**2)

f, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 4))
ax1.plot(x1dot,'r',label="derivative of hare(prediction)")
ax1.plot(x1_pre2,'b',label="derivative of hare(real)")
ax1.legend()
ax2.plot(x2dot,'r',label="derivative of hare(prediction)")
ax2.plot(x2_pre2,'b',label="derivative of lynx(real)")
ax2.legend()
f.suptitle('Comparison of derivatives using Ridge')
f.savefig('derivative_ridge.jpg')




def predict(x,y,n,r1,r2,xi,yi,dt):
    x_pre = np.zeros(n)
    y_pre = np.zeros(n)
    x_pre[0] = x
    y_pre[0] = y
    for i in range(n-1):
        x = x+sum(LAB(x,y)*r1*xi)*dt
        y = y+sum(LAB(x,y)*r2*yi)*dt
        x_pre[i+1] = x
        y_pre[i+1] = y
    return (x_pre,y_pre)
    

x1_pre_1,x2_pre_1 = predict(20,32,30,r11,r12,xi11_new,xi12_new,2)
x1_pre_2,x2_pre_2 = predict(20,32,30,r21,r22,xi21_new,xi22_new,2)
x1_pre_3,x2_pre_3 = predict(20,32,30,r31,r32,xi31_new,xi32_new,2)

plt.figure()
plt.plot(t,x1_pre_1,'r',label='pinv')
plt.plot(t,x1_pre_2,'g',label='LASSO')
plt.plot(t,x1_pre_3,'m',label='Ridge')
plt.plot(t,x1,'b')
plt.legend()
plt.xlabel('year')
plt.ylabel('hare population')
plt.title('prediction and true population for hare')
plt.savefig('pre_h.jpg')
plt.figure()
plt.plot(t,x2_pre_1,'r',label='pinv')
plt.plot(t,x2_pre_2,'g',label='LASSO')
plt.plot(t,x2_pre_3,'m',label='Ridge')
plt.plot(t,x2,'b')
plt.legend()
plt.xlabel('year')
plt.ylabel('lynx population')
plt.title('prediction and true population for lynx')
plt.savefig('pre_l.jpg')




#def vdp1(t, y,a,b,xi,yi):
#    return np.array([sum(LAB(y[0],y[1])[a]*xi[a]), 
#                     sum(LAB(y[0],y[1])[b]*yi[b])])
#t0, t1 = 1, 30                # start and end
#tspan = np.linspace(t0, t1, 30)  # the points of evaluation of solution
#y0 = [20, 32]                   # initial value
#y = np.zeros((len(tspan), len(y0)))   # array for solution
#y[0, :] = y0
#r = integrate.ode(vdp1).set_integrator("dopri5")  # choice of method
#r.set_initial_value(y0, t0)   # initial values
#for i in range(1, tspan.size):
#   y[i, :] = r.integrate(tspan[i]) # get one more value, add it to the array
#   if not r.successful():
#       raise RuntimeError("Could not integrate")
#
#
#
## solve pde
#y0 = [20,32]
#tspan = [1845,1903]
#def vdp1(t, y):
#    return np.array([sum(LAB(y[0],y[1])[a1]*xi11[a1]), 
#                     sum(LAB(y[0],y[1])[b1]*xi12[b1])])
#def vdp2(t, y):
#    return np.array([sum(LAB(y[0],y[1])[a2]*xi21[a2]), 
#                     sum(LAB(y[0],y[1])[b2]*xi22[b2])])
#def vdp3(t, y):
#    return np.array([sum(LAB(y[0],y[1])[a3]*xi31[a3]), 
#                     sum(LAB(y[0],y[1])[b3]*xi32[b3])])


#sol1 = solve_ivp(vdp1,tspan,y0,method='RK45')
#sol2 = solve_ivp(vdp2,tspan,y0,method='RK45')
#sol3 = solve_ivp(vdp3,tspan,y0,method='RK45')
#plt.figure(9)
#plt.plot(np.linspace(1845,1903,len(sol1.y[0,:])),sol1.y[0,:],'r')
#plt.plot(np.linspace(1845,1903,len(sol2.y[0,:])),sol2.y[0,:],'g')
#plt.plot(np.linspace(1845,1903,len(sol3.y[0,:])),sol3.y[0,:],'y')
#plt.plot(t,x1,'b')
#plt.figure(10)
#plt.plot(np.linspace(1845,1903,len(sol1.y[0,:])),sol1.y[1,:],'r')
#plt.plot(np.linspace(1845,1903,len(sol2.y[0,:])),sol2.y[1,:],'g')
#plt.plot(np.linspace(1845,1903,len(sol3.y[0,:])),sol3.y[1,:],'y')
#plt.plot(t,x2,'b')




# KL divergence
hist_h = np.histogram(x1,4)
hist_l = np.histogram(x2,4)
hist11 = np.histogram(x1_pre_1,4)
hist12 = np.histogram(x2_pre_1,4)
hist21 = np.histogram(x1_pre_2,4)
hist22 = np.histogram(x2_pre_2,4)
hist31 = np.histogram(x1_pre_3,4)
hist32 = np.histogram(x2_pre_3,4)
kl11 = scipy.stats.entropy(hist11[0], hist_h[0])
kl12 = scipy.stats.entropy(hist12[0], hist_l[0])
kl21 = scipy.stats.entropy(hist21[0], hist_h[0])
kl22 = scipy.stats.entropy(hist22[0], hist_l[0])
kl31 = scipy.stats.entropy(hist31[0], hist_h[0])
kl32 = scipy.stats.entropy(hist32[0], hist_l[0])
'''
# KL Divergence
def KL(x1,x2,y1,y2,threshold1,threshold2):
    pre_l = len(x1)
    pre_r = len(x1)
    I1 = 0
    II1 = 0
    III1 = 0
    IV1 = 0
    I2 = 0
    II2 = 0
    III2 = 0
    IV2 = 0
    KL = 0
    for i in range(len(x1)):
        if x1[i] > threshold1 and x2[i] > threshold2:
            I1 += 1
        if x1[i] < threshold1 and x2[i] > threshold2:
            II1 += 1
        if x1[i] < threshold1 and x2[i] < threshold2:
            III1 += 1
        if x1[i] > threshold1 and x2[i] < threshold2:
            IV1 ++ 1
        if y1[i] > threshold1 and y2[i] > threshold2:
            I2 += 1
        if y1[i] < threshold1 and y2[i] > threshold2:
            II2 += 1
        if y1[i] < threshold1 and y2[i] < threshold2:
            III2 += 1
        if y1[i] > threshold1 and y2[i] < threshold2:
            IV2 += 1
    p1 = I1/pre_l
    p2 = II1/pre_l
    p3 = III1/pre_l
    p4 = IV1/pre_l
    q1 = I2/pre_r
    q2 = II2/pre_r
    q3 = III2/pre_r
    q4 = IV2/pre_r
    p = np.array([p1,p2,p3,p4])
    q = np.array([q1,q2,q3,q4])
    for i in range(4):
        if p[i] != 0:
            KL += p[i]*(np.log(p[i]/q[i]))
    return KL
plt.figure()
plt.plot(x1_pre_1,x2_pre_1,'ro',label = 'prediction')
plt.plot(x1,x2,'bo',label = 'true value')
plt.xlabel('hare')
plt.ylabel('lynx')
plt.title('hare vs lynx from pinv model')
plt.legend()
plt.savefig('predictions1.jpg')
plt.figure()
plt.plot(x1_pre_2,x2_pre_2,'ro',label = 'prediction')
plt.plot(x1,x2,'bo',label = 'true value')
plt.xlabel('hare')
plt.ylabel('lynx')
plt.title('hare vs lynx from LASSO model')
plt.legend()
plt.savefig('predictions2.jpg')
plt.figure()
plt.plot(x1_pre_3,x2_pre_3,'ro',label = 'prediction')
plt.plot(x1,x2,'bo',label = 'true value')
plt.xlabel('hare')
plt.ylabel('lynx')
plt.title('hare vs lynx from ridge model')
plt.legend()
plt.savefig('predictions3.jpg')

kl1 = KL(x1_pre_1,x2_pre_1,x1,x2,80,40)
kl2 = KL(x1_pre_2,x2_pre_2,x1,x2,20,40)
kl3 = KL(x1_pre_3,x2_pre_3,x1,x2,80,40)
'''            




# AIC
def AIC(y,y_pred,k):
    res = y - y_pred
    sse = sum(res**2)
    a = 2*k-2*np.log(sse)
    return a
# k is the number of predictors

AIC11 = AIC(x1_pre_1,x1,len(a1[0]))
AIC12 = AIC(x2_pre_1,x2,len(b1[0]))
AIC21 = AIC(x1_pre_2,x1,len(a2[0]))
AIC22 = AIC(x2_pre_2,x2,len(b2[0]))
AIC31 = AIC(x1_pre_3,x1,len(a3[0]))
AIC32 = AIC(x2_pre_3,x2,len(b3[0]))




# BIC
def BIC(y,y_pred,k,n):
    res = y - y_pred
    sse = sum(res**2)
    b = np.log(n)*k - 2*np.log(sse)
    return b
# n is the number of data

BIC11 = BIC(x1_pre_1,x1,len(a1[0]),len(x1_pre_1))
BIC12 = BIC(x2_pre_1,x2,len(b1[0]),len(x2_pre_1))
BIC21 = BIC(x1_pre_2,x1,len(a2[0]),len(x1_pre_2))
BIC22 = BIC(x2_pre_2,x2,len(b2[0]),len(x2_pre_2))
BIC31 = BIC(x1_pre_3,x1,len(a3[0]),len(x1_pre_3))
BIC32 = BIC(x2_pre_3,x2,len(b3[0]),len(x2_pre_3))


# Delay time embeding
X = np.array([x1_new,x2_new])
h = hankel(X)
[u,s,v] = np.linalg.svd(h)
plt.figure()
plt.plot(s/sum(s),'o')
plt.title('Singular values')
plt.savefig('sig.jpg')

