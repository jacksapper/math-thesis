# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def step_size(u, v, tech='dynamic', size=.00001):
    if tech=='dynamic':
        return np.dot(u,v)/np.dot(v,v)
    elif tech=='static':
        return size
    
def f(h, n, u):
    result = 0
    for i in range(1,n+1):
        result += ( ((u[i] - u[i-1])/(h)) - ((u[i] + u[i-1]) / 2.) )**2
    result *= (h/2.)
    return result
    
def df(h, n, u):
    result = np.ones(n+1)
    for i in range(1,n):
        result[i] = (((u[i] - u[i-1])/(h)) - ((u[i] + u[i-1]) / 2.))*((1/h)-(1/2.)) + (((u[i+1] - u[i])/(h)) - ((u[i+1] + u[i]) / 2.))*((-1/h)-(1/2.))
    result[0] = (((u[i+1] - u[i])/(h)) - ((u[i+1] + u[i]) / 2.))*((-1/h)-(1/2.))
    result[n] = (((u[i] - u[i-1])/(h)) - ((u[i] + u[i-1]) / 2.))*((1/h)-(1/2.))
    return result
    
def graph(x,y1,y2):
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.show()
    
#Constants.  This defines intervals of equal distance on [lbound,ubound]
LBOUND = 0.
UBOUND = 1.
INTERVALS = 500
EPSILON = 10**-3


#Set intervals as linearly spaced and the initial guess for y
x = np.linspace(LBOUND,UBOUND,INTERVALS+1)
interval_length = abs(x[1]-x[0])
yold = np.zeros(INTERVALS+1)
ynew = 1. * np.ones(INTERVALS+1)
k = 0

while f(interval_length,INTERVALS,ynew) > EPSILON:
    grad = df(interval_length, INTERVALS, ynew)
    s = step_size(ynew,grad)
    yold = np.copy(ynew)
    ynew = yold - s*grad
    if k%100 == 0:
        print(k, f(interval_length,INTERVALS,ynew))
    k=k+1
yexact = np.exp(x)

graph(x,ynew,yexact)
print(k)