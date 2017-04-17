# -*- coding: utf-8 -*-
#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
#---CONSTANTS---
BOUNDS = (0,10)
POINTS = 2**7
EPSILON = 10**-10
INITIAL = (1,4)
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
#y = np.random.rand(POINTS)
y = np.ones(POINTS)
proj = np.ones(POINTS)
#Sets the initial condition and 
#then defines a projection vector from df 
#to make sure it stays fixed.
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

sys0f =  lambda t,u:  D1 @ u + D0 @ (u**2)
sys0df = lambda t,u:  D1 + 2*D0*u
sys1f =    lambda t,u:  D1 @ u - t*np.exp(3*t) - 2*D0@u
sys1df =   lambda t,u:  D1 - np.exp(3*t) - 3*t*np.exp(3*t) -2*D0
sys2f =    lambda t,u:  D1@y - D0 @ np.cos(2*t) - D0 @ np.sin(3*t)
sys2df =   lambda t,u:  D1 + D0 * np.sin(2*t) + D0 * np.cos(3*t)


f = sys2f
df = sys2df

phi =       lambda f:       0.5*f.dot(f)
dphi =      lambda f, df:   proj * (df.T @ f)
sobolev =   lambda u:       (lin.solve(A,u))

def graph(x,y):
    ax1.clear()
    ax1.plot(x,y)
    plt.plot(x,y)
    fig.canvas.draw()
    
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x,y)
plt.plot()
fig.canvas.draw()

k = 0
while phi(f(x,y)) > EPSILON or not np.isfinite(phi(f(x,y))):
    grad =  sobolev(dphi( f(x,y), df(x,y) ))
    s = step_nonlin(x,grad)
    #s = 0.00001
    y -= s*grad
    if k%2**0 == 0:
        print(k, phi(f(x,y)))
        plt.plot(x,y)
        plt.show()
    k=k+1
