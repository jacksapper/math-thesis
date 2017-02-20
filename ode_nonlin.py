# -*- coding: utf-8 -*-
#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
#---CONSTANTS---
BOUNDS = (-1.,1.)
POINTS = 2**6
EPSILON = 10**-16
INITIAL = None
STEP_SIZE = 0.001
#---DERIVED---DON'T CHANGE THESE, CHANGE THE REAL CONSTANTS
INTERVAL_LENGTH = (BOUNDS[1] - BOUNDS[0])/(POINTS-1)
D0 = .5*(np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
D1 = (1/INTERVAL_LENGTH)*(-1*np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
A = D1.T @ D1 + D0.T @ D0
x = np.linspace(BOUNDS[0], BOUNDS[1],POINTS)
y = np.random.rand(POINTS)
proj = np.ones(POINTS)
#---Sets the initial condition and then defines a projection vector from df to make sure it stays fixed.
if INITIAL is not None:
    fixed = np.argmin(abs(x-INITIAL[0]))
    proj[fixed] = 0
    y[fixed] = INITIAL[1]

step_lin =      lambda u, v:    u.dot(v)/v.dot(v)
step_nonlin =   lambda u, g:    (df(u) @ g).dot(f(u)) / (df(u) @ g).dot(df(u) @ g)
f =             lambda u:       D1 @ u + D0 @ (u**2)
df =            lambda u:       proj * (D1 + 2*D0*u)
phi =           lambda f:       0.5*f.dot(f)
dphi =          lambda f, df:   df.T @ f
sobolev =       lambda u:       lin.solve(A,u)

def graph(x,y):
    plt.plot(x,y)
    plt.show()

k = 0

while phi(f(y)) > EPSILON or not np.isfinite(phi(f(y))):
    grad =  sobolev(dphi( f(y), df(y) ))
    s = step_nonlin(y,grad)
    y -= s*grad
    if k%2 == 0:
        print(k, phi(f(y)))
        graph(x,y)
    k=k+1
print(k)
