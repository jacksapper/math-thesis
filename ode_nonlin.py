# -*- coding: utf-8 -*-
#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
#---CONSTANTS---
LBOUND = 1.
UBOUND = 2.
POINTS = 2**6
EPSILON = 10**-16
INITIAL = None

#Matrix is O(POINTS**2)

#---DERIVED CONSTANTS---DON'T CHANGE THESE, CHANGE THE REAL CONSTANTS
INTERVAL_LENGTH = (UBOUND-LBOUND)/(POINTS-1)
D0 = .5*(np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
D1 = (1/INTERVAL_LENGTH)*(-1*np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
#D = D1 + D0
k = 0
A = D1.T @ D1 + D0.T @ D0

#---FUNCTIONS---        
def step_size(u, v, tech='dynamic', size=EPSILON/10):
    if tech=='dynamic':
        upper = u.dot(v)
        lower = v.dot(v)
    elif tech=='static':
        return size
    
def f(u):
    #f = D1 @ u + D0 @ (u**2)
    f = D1 @ u + (D0 @ u)**2
    result = .5*(f).dot(f) # .5 || f(u) ||**2
    return result
    
def df(u,v):
    L = D1 @ v
    #NL = D0 @ (2*(u*v))
    NL = 2*(D0 @ u)*(D0 @ v)
    grad2 = L + NL
    if INITIAL is not None:
        grad2[index] = 0
    return f(u) * np.append(grad2,grad2[len(grad2)-1])
    
def sobolev(u):
    gradH = lin.solve(A,u)
    if INITIAL is not None:
        gradH[index] = 0
    return gradH
    
def graph(x,y1):
    plt.plot(x,y1)
    plt.show()
    
#---MAIN---
x = np.linspace(LBOUND,UBOUND,POINTS)
yold = np.zeros(POINTS)
ynew = 1*np.ones(POINTS)
#yexact = np.exp(x)
grad = np.random.rand(POINTS)

if INITIAL is not None:
    index = np.argmin(abs(x-INITIAL[0]))
    ynew[index] = INITIAL[1]

while f(ynew) > EPSILON and np.isfinite(f(ynew)):
    grad = (df(ynew,grad))
    s = step_size((ynew),(grad),'static')
    yold = np.copy(ynew)
    ynew = yold - s*grad
    if k%2 == 0:
        print(k, f(ynew))
        graph(x,ynew)
    k=k+1

graph(x,ynew)
print(k)
