# -*- coding: utf-8 -*-
#Code to solve:
#   x' =  [a,b] x + Px
#   y' =  [c,d] y + Py

#TODO: Sobolev + CSR
#TODO: Initial

#---IMPORTS---
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
#---CONSTANTS---
LBOUND  = -100.
UBOUND  = 100.
POINTS  = 2**7
EPSILON = 10**-12
#INITIAL = (t0,x0,y0) or None
INITIAL = None

t = np.mat(np.linspace(LBOUND,UBOUND,POINTS)).T
u = np.mat(np.random.rand(2*POINTS)).T
x = u[:POINTS]
y = u[POINTS:]

a = -1/3
b = -2/3
c = -2/3
d = -1/3

display = [1,10,100,1000,10000,100000]
#---DERIVED CONSTANTS---DON'T CHANGE THESE, CHANGE THE REAL CONSTANTS
INTERVAL_LENGTH = (UBOUND-LBOUND)/(POINTS-1)
I = np.mat(np.eye(POINTS-1,POINTS))

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
def f(u):
    result = .5*(D*u).T*(D*u)
    return .5*result[0,0] 
    
def df(u): 
    grad2 = D.T*D*u
    if INITIAL is not None:
        grad2[index,0] = 0
        grad2[index+POINTS,0] = 0
    return grad2
    
def sobolev(u):
    gradH = spsolve(A,u) 
    if INITIAL is not None:
        gradH[index] = 0
    return np.mat(gradH).T
    
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
if INITIAL is not None:
    index = np.argmin(abs(t-INITIAL[0]))
    x[index] = INITIAL[1]
    y[index] = INITIAL[2]
    for i in range(0,POINTS):
        u[i] = INITIAL[1]
        u[i+POINTS] = INITIAL[2]

k = 0

#plt.gca(projection='3d')
#plt.xlim(-1.2,1.2)
#plt.ylim(-1.2,1.2)
while f(u) > EPSILON and np.isfinite(f(u)):
    grad = sobolev(df(u))
    s = step_size(D*u,D*grad,tech='dynamic')
    u -= s*grad
    k=k+1
    if k in display:
        print("|",k,"|",
        round(f(u),3), "|",
        round(s,6),"|")
        #plt.plot(x.A1,y.A1,t.A1,label='k={num}'.format(num=k))
        
		  
        
print(k)
#plt.legend(loc='lower right')
#plt.savefig(
#'/home/jason/Dropbox/thesis/'
#'newimg/circles-3d.png',dpi=150)
#plt.display()