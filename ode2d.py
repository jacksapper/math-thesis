# -*- coding: utf-8 -*-
#Code to solve:
#   x' = y
#   y' = -x


#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
#---CONSTANTS---
LBOUND  = -10.
UBOUND  = 10.
POINTS  = 50
EPSILON = 10**-3
INITIAL = (0,1,1)
#INITIAL = (t0,x0,y0) or None
a = 1
b = 2
c = -2
d = 1

#---DERIVED CONSTANTS---DON'T CHANGE THESE, CHANGE THE REAL CONSTANTS
INTERVAL_LENGTH = (UBOUND-LBOUND)/(POINTS-1)

D0   = np.asmatrix(np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1)) / 2
D1   = np.asmatrix(-1*np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1)) / INTERVAL_LENGTH
ZERO = np.zeros(D0.shape)


D    = csr_matrix(np.concatenate(( np.concatenate( (D1-a*D0, -c*D0  ), axis=0) , np.concatenate( (-b*D0, D1-d*D0 ), axis=0)),axis=1))
A0   = csr_matrix(np.concatenate(( np.concatenate( (D0, ZERO), axis=0) , np.concatenate( (ZERO, D0), axis=0)),axis=1))
A1   = csr_matrix(np.concatenate(( np.concatenate( (D1, ZERO), axis=0) , np.concatenate( (ZERO, D1), axis=0)),axis=1))

A    = A0.T * A0 + A1.T * A1
#A = A0.T * A0 + D.T * D

#---FUNCTIONS---
def f(u):
    result = (D*u).T*(D*u)
    return .5*result[0,0] 
    
def df(u):
    grad2 = D.T*D*u
    if INITIAL is not None:
        grad2[index,0] = 0
        grad2[index+POINTS,0] = 0
    return grad2
    
def step_size(u, v, tech='dynamic', size=5*EPSILON):
    if tech=='dynamic':
        upper = u.T*v
        lower = v.T*v
        return upper[0,0]/lower[0,0]
    elif tech=='static':
        return size

def graph(x,y1,y2=None):
    plt.plot(x,y1)
    if y2 is not None:
        plt.plot(x,y2)
        
def graph3d(x,y,t):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x.A1, y.A1, t.A1, label='parametric curve')
    ax.legend()
    plt.show()
    
def save_graph(x,y):
    plt.plot(x,y)
    plt.savefig('iter'+str(k)+'.png')
    
def save_graph3d(x,y,t):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x.A1, y.A1, t.A1, label='parametric curve')
    ax.legend()
    plt.savefig('iter'+str(k)+'.png')
    
#---MAIN---
t = np.asmatrix(np.linspace(LBOUND,UBOUND,POINTS)).T
u_old = np.asmatrix(np.zeros(2*POINTS)).T
#u = np.asmatrix(np.ones(2*POINTS)).T
u = np.asmatrix(20*np.random.rand(2*POINTS) - 10).T
x = u[:POINTS]
y = u[POINTS:]

if INITIAL is not None:
    index = np.argmin(abs(t-INITIAL[0]))
    x[index] = INITIAL[1]
    y[index] = INITIAL[2]

k = 0
while f(u) > EPSILON and np.isfinite(f(u)):
    grad = df(u)
    s = step_size(D*u,D*grad,tech='static')
    u_old = np.copy(u)
    u -= s*grad
    k=k+1
    if k%100 == 0:
        print(k, f(u))
    if k%100 == 0:
        graph3d(x,y,t)




