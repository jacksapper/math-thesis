# -*- coding: utf-8 -*-

#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt

#---CONSTANTS---
LBOUND = 0.
UBOUND = 2.
POINTS = 50
EPSILON = 10**-3

#---DERIVED CONSTANTS---DON'T CHANGE THESE, CHANGE THE REAL CONSTANTS
INTERVAL_LENGTH = (UBOUND-LBOUND)/(POINTS-1)
D0 = .5*np.asmatrix(np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
D1 = (1/INTERVAL_LENGTH)*np.asmatrix(-1*np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
D = D1-D0


#---FUNCTIONS---
def step_size(u, v, tech='dynamic', size=.00001):
    if tech=='dynamic':
        upper = u*v.T
        lower = v*v.T
        return upper[0,0]/lower[0,0]
    elif tech=='static':
        return size
    
def f(h,n,u):
    result = (D*ynew.T).T*(D*ynew.T)
    return .5*result[0,0] 
    
#u and v need to be column matrices not arrays for the * operator to work correctly
def df(h, n, u):
    A=(D.T*D)*u.T
    return A.T
    
def graph(x,y1,y2):
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.show()
    
#---MAIN---
x = np.asmatrix(np.linspace(LBOUND,UBOUND,POINTS))
yold = np.asmatrix(np.zeros(POINTS))
ynew = np.asmatrix(4. * np.ones(POINTS))
yexact = np.exp(x)

k = 0
while f(INTERVAL_LENGTH,POINTS-1,ynew) > EPSILON:
    grad = df(INTERVAL_LENGTH, POINTS, ynew)
    s = step_size(ynew,grad,'dynamic')
    yold = np.copy(ynew)
    ynew = yold - s*grad
    if k%1000 == 0:
        print(k, f(INTERVAL_LENGTH,POINTS,ynew))
    if k%1000 == 0:
        graph(x.T,ynew.T,yexact.T)
    k=k+1

graph(x.T,ynew.T,yexact.T)
print(k)
