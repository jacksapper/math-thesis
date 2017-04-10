#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:58:51 2017

@author: jason
"""
#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#---MODIFIERS---
BOUNDS          = (-10,10)
POINTS          = 2**6
EPSILON         = 2**-15
INITIAL         = None
(a,b,c,d)       = (0,1,-1,0)
#---CONSTANTS---
INTERVAL_LENGTH = (UBOUND-LBOUND)/(POINTS-1)
D0              = np.asmatrix(np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1)) / 2
D1              = np.asmatrix(-1*np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1)) / INTERVAL_LENGTH
A               = D1.T @ D1 + D0.T @ D0
proj            = np.ones(POINTS)
#---SYSTEMS---
linear          = (lambda x,y: a*D0@x + b*D0@y - D1@x, lambda x,y: a*D0 - D1, lambda x,y: b*D0@y,
                   lambda x,y: c*D0@x + d*D0@y - D1@y, lambda x,y: c*D0, lambda x,y: d*D0 - D1)
#---FUNCTIONS---
(f,dfx,dfy,g,dgx,dgy)     = linear
sobolev         = lambda u: np.linalg.lin.solve(A,u)
phi             = lambda f,g: 0.5*(f.dot(f)+g.dot(g))
dphi            = lambda f,df,g,dg: df.T@f + dg.T@g

k = 0
while phi(f(x,y),g(x,y)) > EPSILON or not np.isfinite(phi(f(x,y),g(x,y))):
    gradx = dphi(f(x,y),g(x,y))
    s = 0.001
    x -= s*grad
    y -= s*grad
    if k%2**3 == 0:
        print(k)
        plt.plot(x,y)
        plt.show()
    k+=1

