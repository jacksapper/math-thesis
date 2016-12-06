# -*- coding: utf-8 -*-
#Code to solve:
#   x' =  [a,b] x
#   y' =  [c,d] y

#TODO: Sobolev + CSR
#TODO: Initial

#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
#---CONSTANTS---
LBOUND  = -5.
UBOUND  = 5.
POINTS  = 2**7
EPSILON = 10**-12
INITIAL = (0,2,3)
#INITIAL = (t0,x0,y0) or None
a = 0
b = 1
c = -4
d = 0

#---DERIVED CONSTANTS---DON'T CHANGE THESE, CHANGE THE REAL CONSTANTS
INTERVAL_LENGTH = (UBOUND-LBOUND)/(POINTS-1)

D0   = np.mat(np.eye(POINTS-1,POINTS) \
+ np.roll(np.eye(POINTS-1,POINTS),1,1)) / 2
D1   = np.mat(-1*np.eye(POINTS-1,POINTS) \
+ np.roll(np.eye(POINTS-1,POINTS),1,1)) / INTERVAL_LENGTH
ZERO = np.zeros(D0.shape)


D    = csr_matrix(np.concatenate(( np.concatenate( (D1-a*D0, -c*D0  ), axis=0),
                  np.concatenate( (-b*D0, D1-d*D0 ), axis=0)),axis=1))
A0   = csr_matrix(np.concatenate(( np.concatenate( (D0, ZERO), axis=0),
                  np.concatenate( (ZERO, D0), axis=0)),axis=1))
A1   = csr_matrix(np.concatenate(( np.concatenate( (D1, ZERO), axis=0),
                  np.concatenate( (ZERO, D1), axis=0)),axis=1))

A    = A0.T * A0 + A1.T * A1
#A = A0.T * A0 + D.T * D


#---FUNCTIONS---
def f(u): #35 µs
    result = .5*(D*u).T*(D*u)
    return .5*result[0,0] 
    
def df(u): #210 µs
    grad2 = D.T*D*u
    if INITIAL is not None:
        grad2[index,0] = 0
        grad2[index+POINTS,0] = 0
    return grad2
    
def sobolev(u): #97.1 µs
    gradH = spsolve(A,u) #91.1 µs
    if INITIAL is not None:
        gradH[index] = 0
    return np.mat(gradH).T
    
def step_size(u, v, tech='dynamic', size=1*EPSILON): #36 µs with Du and dynamic
    if tech=='dynamic':
        upper = u.T*v
        lower = v.T*v
        return upper[0,0]/lower[0,0]
    elif tech=='static':
        return size

def graph(x,y1,y2=None): #670 µs
    plt.plot(x,y1)
    if y2 is not None:
        plt.plot(x,y2)
        
def graph3d(x,y,t): #83 ms or 83,000 µs
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x.A1, y.A1, t.A1, label='parametric curve')
    ax.legend()
    plt.show()
    
def save_graph(x,y): #37.8 ms or 37,800 µs
    plt.plot(x,y)
    plt.savefig('iter'+str(k)+'.png')
    
def save_graph3d(x,y,t): #73.5 ms or 73,500 µs
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x.A1, y.A1, t.A1, label='parametric curve')
    ax.legend()
    plt.savefig('iter'+str(k)+'.png')
    
#---MAIN---
t = np.mat(np.linspace(LBOUND,UBOUND,POINTS)).T
u = np.mat(np.ones(2*POINTS)).T
x = u[:POINTS]
y = u[POINTS:]

if INITIAL is not None:
    index = np.argmin(abs(t-INITIAL[0]))
    x[index] = INITIAL[1]
    y[index] = INITIAL[2]
    for i in range(0,POINTS):
        u[i] = INITIAL[1]
        u[i+POINTS] = INITIAL[2]

k = 0
while f(u) > EPSILON and np.isfinite(f(u)):
    grad = sobolev(df(u)) #df takes 210 µs, df+sobolev 342 µs
    s = step_size(D*u,D*grad,tech='dynamic') #36 µs
    u -= s*grad #4.27 µs
    k=k+1
    if k%2**9 == 0:
        print(k,f(u)) #computing f(u) again takes 35.9 µs
        graph3d(x,y,t) #82.9 ms 

graph3d(x,y,t)
graph(x,y)



