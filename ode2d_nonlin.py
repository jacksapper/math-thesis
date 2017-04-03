#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:21:29 2017

@author: jason
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BOUNDS = (-1,1)
POINTS = 2**6
EPSILON = 2**-16
UPPER = 2**32
INTERVAL_LENGTH = (BOUNDS[1]-BOUNDS[0])/(POINTS - 1)

d0   = np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1) / 2
d1   = -1*np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1) / INTERVAL_LENGTH
ZERO = np.zeros(d0.shape)
D0   = (np.concatenate(( np.concatenate( (d0, ZERO), axis=0) , np.concatenate( (ZERO, d0), axis=0)),axis=1))
D1   = (np.concatenate(( np.concatenate( (d1, ZERO), axis=0) , np.concatenate( (ZERO, d1), axis=0)),axis=1))
A    = D0.T @ D0 + D1.T @ D1


def graph3d(x,y,t):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, t, 'r--', label='parametric curve')
    ax.legend()
    plt.show()

f =             lambda u:          D1 @ (x*u) - D0 @ (y*u)
df =            lambda u:          x*(D1) + y*(-D0) #wrong
g =             lambda u:          D1 @ (y*u) + D0 @ (x*u)
dg =            lambda u:          x*(D0) + y*(D1) #wrong
phi =           lambda f,g:          0.5*(f.dot(f)+g.dot(g))
dphi =          lambda f,df,g,dg:    df.T @ f + dg.T @ g
sobolev =       lambda u:            np.linalg.solve(A,u)

t = np.linspace(BOUNDS[0],BOUNDS[1],POINTS)
u = 20*np.random.rand(2*POINTS) - 10
x = np.concatenate([np.ones(POINTS),np.zeros(POINTS)])
y = np.concatenate([np.zeros(POINTS),np.ones(POINTS)])

k = 0
while UPPER > phi(f(u), g(u)) > EPSILON:
    grad = (dphi(f(u),df(u),g(u),dg(u)))
    s = 0.0001
    u -= s*grad
    k+=1
    if (k%2**0) == 0:
        print(k,phi(f(u),g(u)))
        plt.plot(u[POINTS:],u[:POINTS])
        plt.show()