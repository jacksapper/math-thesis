# -*- coding: utf-8 -*-
#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt
#---CONSTANTS---
LBOUND = 0.
UBOUND = 1.
POINTS = 3
EPSILON = 10**-16
INITIAL = None

#Matrix is O(POINTS**2)

#---DERIVED CONSTANTS---DON'T CHANGE THESE, CHANGE THE REAL CONSTANTS
INTERVAL_LENGTH = (UBOUND-LBOUND)/(POINTS-1)
D0 = .5*(np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
D1 = (1/INTERVAL_LENGTH)*(-1*np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
D = D1-D0
A = D1.T @ D1 + D0.T @ D0
k = 0

#---FUNCTIONS---        
def step_size(u, v, tech='dynamic', size=EPSILON/10):
    if tech=='dynamic':
        upper = u.dot(v)
        lower = v.dot(v)
        return upper/lower
    elif tech=='static':
        return size
    
def f(u):
    f = D @ u
    result = (f).dot(f)
    return .5*result
    
#u and v need to be column matrices not arrays for the * operator to work correctly
def df(u):
    grad2=(D.T @ D) @ u
    if INITIAL is not None:
        grad2[index] = 0
    return grad2
    
def sobolev(u):
    gradH = np.linalg.solve(A,u)
    if INITIAL is not None:
        gradH[index] = 0
    return gradH
    
def graph(x,y1,y2):
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.show()
    
#---MAIN---
x = np.linspace(LBOUND,UBOUND,POINTS).T
yold = np.zeros(POINTS).T
ynew = 12. * np.ones(POINTS).T
yexact = np.exp(x)

if INITIAL is not None:
    index = np.argmin(abs(x-INITIAL[0]))
    ynew[index] = INITIAL[1]

while f(ynew) > EPSILON:
    grad = sobolev(df(ynew))
    s = step_size((D @ ynew),(D @ grad),'dynamic')
    yold = np.copy(ynew)
    ynew = yold - s*grad
    if k%10 == 0:
        print(k, f(ynew))
    if k%1000 == 0:
        graph(x,ynew,yexact)
    k=k+1

graph(x,ynew,yexact)
print(k)
