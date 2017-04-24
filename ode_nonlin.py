# -*- coding: utf-8 -*-
#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
#---CONSTANTS---
BOUNDS = (0,10)
POINTS = 2**6
FINAL_POINTS = 2**10
EPSILON = 10**-10
INFINITY = 2**32
INITIAL = None
STEP_SIZE = 0.001
STEP_BOUND = 1

#---DERIVED---DON'T CHANGE THESE, CHANGE THE REAL CONSTANTS

INTERVAL_LENGTH = (BOUNDS[1] - BOUNDS[0])/(POINTS-1)
D0 = .5*(np.eye(POINTS-1,POINTS) \
+ np.roll(np.eye(POINTS-1,POINTS),1,1))
D1 = (1/INTERVAL_LENGTH)*(-1*np.eye(POINTS-1,POINTS) \
+ np.roll(np.eye(POINTS-1,POINTS),1,1))
A = D1.T @ D1 + D0.T @ D0
x = np.linspace(BOUNDS[0], BOUNDS[1],POINTS)
y = np.random.rand(POINTS)
#y = np.ones(POINTS)
proj = np.ones(POINTS)
if INITIAL is not None:
    fixed = np.argmin(abs(x-INITIAL[0]))
    proj[fixed] = 0
    y[fixed] = INITIAL[1]
    A[fixed,:] = 0
    A[fixed,fixed] = 1
    y = INITIAL[1]*np.ones(POINTS)

step_lin = lambda u, v:    u.dot(v)/v.dot(v)
step_nonlin = lambda u, g:\
(df(x,u) @ g).dot(f(x,u)) / (df(x,u) @ g).dot(df(x,u) @ g)

sf0= lambda t,y:  D1 @ y + D0 @ (y**2)
df0= lambda t,y:  D1 + 2*D0*y
sf1= lambda t,y:  D1 @ y - D0@(t+y)
df1= lambda t,y:  D1 - D0
sf2= lambda t,y:  D1@y - D0 @ np.cos(2*t) - D0 @ np.sin(3*t)
df2= lambda t,y:  D1 + D0 * np.sin(2*t) + D0 * np.cos(3*t)
sf3= lambda t,y:  D1 @ y - D0@(np.exp(-t)*np.sin(t))
df3= lambda t,y:  D1 - D0*(-np.exp(-t)*np.sin(t) + np.cos(t)*np.exp(-t))



f = sf3
df = df3

phi =       lambda f:       0.5*f.dot(f)
dphi =      lambda f, df:   proj * (df.T @ f)
sobolev =   lambda u:       (lin.solve(A,u))
    
k = 0

while POINTS <= FINAL_POINTS:
    while EPSILON < phi(f(x,y)) < INFINITY:
        grad = sobolev(dphi(f(x,y),df(x,y)))
        #s = step_nonlin(x,grad)
        s = 10**-1
        y -= s*grad
        if k%2**5 == 0:
            print(k,phi(f(x,y)))
            plt.plot(x,y)
            plt.show()
        k=k+1
    c = np.empty((y.size + (D0 @ y).size,), dtype=y.dtype)
    c[0::2] = y
    c[1::2] = D0 @ y
    y = c
    POINTS = 2*POINTS - 1
    INTERVAL_LENGTH = (BOUNDS[1] - BOUNDS[0])/(POINTS-1)
    D0 = .5*(np.eye(POINTS-1,POINTS) \
    + np.roll(np.eye(POINTS-1,POINTS),1,1))
    D1 = (1/INTERVAL_LENGTH)*(-1*np.eye(POINTS-1,POINTS) \
    + np.roll(np.eye(POINTS-1,POINTS),1,1))
    A = D1.T @ D1 + D0.T @ D0
    x = np.linspace(BOUNDS[0], BOUNDS[1],POINTS)
    proj = np.ones(POINTS)
    if INITIAL is not None:
        fixed = np.argmin(abs(x-INITIAL[0]))
        proj[fixed] = 0
        y[fixed] = INITIAL[1]
        A[fixed,:] = 0
        A[fixed,fixed] = 1

