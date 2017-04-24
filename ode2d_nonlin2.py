#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 09:20:18 2017

@author: jason
"""

# -*- coding: utf-8 -*-
#Code to solve:
#   x' =  [a,b] x
#   y' =  [c,d] y
#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#---CONSTANTS---
LBOUND  = -10.
UBOUND  = 10.
POINTS  = 2**7
EPSILON = 10**-9
INITIAL = None
#INITIAL = (t0,x0,y0) or None

INTERVAL_LENGTH = (UBOUND-LBOUND)/(POINTS-1)
D0 = .5*(np.eye(POINTS-1,POINTS) \
+ np.roll(np.eye(POINTS-1,POINTS),1,1))
D1 = (1/INTERVAL_LENGTH)*(-1*np.eye(POINTS-1,POINTS) \
+ np.roll(np.eye(POINTS-1,POINTS),1,1))
ZERO = np.zeros(D0.shape)

a = lambda x,y: 0
b = lambda x,y: 0
c = lambda x,y: 0
d = lambda x,y: 0
F = lambda x,y: 0
G = lambda x,y: 0

A0 = np.concatenate((np.concatenate((D0,ZERO),axis=0),
     np.concatenate((ZERO,D0),axis=0)),axis=1)
A1 = np.concatenate((np.concatenate((D1,ZERO),axis=0),
     np.concatenate((ZERO,D1),axis=0)),axis=1)
A  = A0.T @ A0 + A1.T @ A1
#A = A0.T * A0 + D.T * D

#---FUNCTIONS---
def phi(u):
    tmp = f(u)
    return 0.5 * tmp.dot(tmp)

def f(u):
    return A1 @ u - np.concatenate(
    (D0 @ (F(x,y)), D0 @ G(x,y)),axis=0)

def fprime(u):
    return np.concatenate(( 
    np.concatenate( (D1-a(x,y)*D0, -c(x,y)*D0  ), axis=0),
    np.concatenate( (-b(x,y)*D0, D1-d(x,y)*D0 ), axis=0)),
    axis=1)

def df(u):
    grad2 = fprime(u).T @ f(u)
    if INITIAL is not None:
        grad2[index,0] = 0
        grad2[index+POINTS,0] = 0
    return grad2

def sobolev(u):
    gradH = np.linalg.solve(A,u)
    if INITIAL is not None:
        gradH[index,0] = 0
        gradH[index+POINTS,0] = 0
    return gradH

def graph(x,y1,y2=None):
    plt.plot(x,y1)
    if y2 is not None:
        plt.plot(x,y2)

def graph3d(x,y,t):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.plot(x, y, t, label='parametric curve')
    ax.legend()
    plt.show()

def updateD():
    return np.concatenate((
	np.concatenate((D1-a(x,y)*D0, -c(x,y)*D0),axis=0),
    np.concatenate((-b(x,y)*D0, D1-d(x,y)*D0),axis=0)),
    axis=1)

#---MAIN---
t = (np.linspace(LBOUND,UBOUND,POINTS)).T
u_old = (np.zeros(2*POINTS)).T
u = (.7*np.ones(2*POINTS)).T
#u = (20*np.random.rand(2*POINTS) - 10).T
x = u[:POINTS]
y = u[POINTS:]

if INITIAL is not None:
    index = np.argmin(abs(t-INITIAL[0]))
    x[index] = INITIAL[1]
    y[index] = INITIAL[2]

k = 0
D = updateD()
while phi(u) > EPSILON and np.isfinite(phi(u)):
    grad = sobolev(df(u))
    if k < 20:
        s = 10**-1
    else:
        s = 10**-2
    u -= s*grad
    k=k+1
    if k%2**5 == 0:
        print(k, phi(u))
        #graph3d(x,y,t)
    D = updateD()
    
graph3d(x,y,t)
plt.plot(t,y)
plt.plot(t,x,label='x(t)=y(t)')

plt.legend(loc='lower right')
plt.show()

