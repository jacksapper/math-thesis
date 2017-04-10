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

D0   = np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1) / 2
D1   = -1*np.eye(POINTS-1,POINTS) + np.roll(
        np.eye(POINTS-1,POINTS),1,1) / INTERVAL_LENGTH
ZERO = np.zeros(d0.shape)
A    = D0.T @ D0 + D1.T @ D1

def graph3d(x,y,t):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, t, 'r--', label='parametric curve')
    ax.legend()
    plt.show()

p =             lambda x,y:     y
q =             lambda x,y:     -x
dpu =           lambda x,y:     0
dpv =           lambda x,y:     1
dqu =           lambda x,y:     -1
dqv =           lambda x,y:     0
phi =           lambda D,u:     .5*(D@u).dot(D@u)
Eg =            lambda D,u:       D.T @ D @ u #wrong
sobolev =       lambda Eg:      np.linalg.solve(A,Eg)

t = np.linspace(BOUNDS[0],BOUNDS[1],POINTS)
u = 20*np.random.rand(2*POINTS) - 10
x = u[POINTS:]
y = u[:POINTS]
D    = (np.concatenate((np.concatenate( (D1-dpu(x,y)*D0, -dqu(x,y)*D0  ),axis=0),
        np.concatenate( (-dpv(x,y)*D0, D1-dqv(x,y)*D0 ), axis=0)),axis=1))

k = 0
while UPPER >  phi(D,u) > EPSILON:
    D    = (np.concatenate((np.concatenate( (D1-dpu(x,y)*D0, -dqu(x,y)*D0  ),axis=0),
        np.concatenate( (-dpv(x,y)*D0, D1-dqv(x,y)*D0 ), axis=0)),axis=1))
    grad = (Eg(D,u))
    s = 0.0001
    u -= s*grad
    k+=1
    if (k%2**0) == 0:
        print(k,phi(D,u))
        graph3d(x,y,t)