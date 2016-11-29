# -*- coding: utf-8 -*-
#Code to solve:
#   x' = y
#   y' = -x


#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt

#---CONSTANTS---
LBOUND  = -100.
UBOUND  = 100.
POINTS  = 100
EPSILON = 10**-3
INITIAL = None

#---DERIVED CONSTANTS---DON'T CHANGE THESE, CHANGE THE REAL CONSTANTS
INTERVAL_LENGTH = (UBOUND-LBOUND)/(POINTS-1)

D0   = .5*np.asmatrix(np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
D1   = (1/INTERVAL_LENGTH)*np.asmatrix(-1*np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
ZERO = np.zeros(D0.shape)


D    = np.concatenate(( np.concatenate( (D1, D0  ), axis=0) , np.concatenate( (-D0, D1 ), axis=0)),axis=1) 
A0   = np.concatenate(( np.concatenate( (D0, ZERO), axis=0) , np.concatenate( (ZERO, D0), axis=0)),axis=1) 
A1   = np.concatenate(( np.concatenate( (D1, ZERO), axis=0) , np.concatenate( (ZERO, D1), axis=0)),axis=1) 

A    = A0.T * A0 + A1.T * A1
#A = A0.T * A0 + D.T * D

#---FUNCTIONS---
def f(u):
    result = (D*u).T*(D*u)
    return .5*result[0,0] 
    
def df(u):
    result = D.T*D*u
    return result

def graph(x,y1,y2=None):
    plt.plot(x,y1)
    if y2 is not None:
        plt.plot(x,y2)
    
#---MAIN---
t = np.asmatrix(np.linspace(LBOUND,UBOUND,POINTS)).T
u_old = np.asmatrix(np.zeros(2*POINTS)).T
u = np.asmatrix(np.ones(2*POINTS)).T
x = u[:POINTS]
y = u[POINTS:]


k = 0
while f(u) > EPSILON:
    grad = df(u)
    s = EPSILON/8
    u_old = np.copy(u)
    u -= s*grad
    k=k+1
    if k%100 == 0:
        print(k, f(u))
    if k%10000 == 0:
        plt.plot(x,y)
        plt.savefig('iter'+str(k)+'.png')




