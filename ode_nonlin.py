# -*- coding: utf-8 -*-
#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
#---CONSTANTS---
LBOUND = -1.
UBOUND = 1.
POINTS = 2**6
EPSILON = 10**-16
INITIAL = None

#Matrix is O(POINTS**2)

#---DERIVED CONSTANTS---DON'T CHANGE THESE, CHANGE THE REAL CONSTANTS
INTERVAL_LENGTH = (UBOUND-LBOUND)/(POINTS-1)
<<<<<<< HEAD
D0 = (.5*(np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1)))
D1 = ((1/INTERVAL_LENGTH)*(-1*np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1)))
=======
D0 = .5*(np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
D1 = (1/INTERVAL_LENGTH)*(-1*np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
>>>>>>> d7586146040a05c90e1c8ca1f578a2e5793b2b90
#D = D1 + D0
k = 0
A = D1.T @ D1 + D0.T @ D0

#---FUNCTIONS---        
def step_size(u, v, tech='dynamic', size=1):
    if tech=='dynamic':
        upper = u.dot(v)
        lower = v.dot(v)
        return upper/lower
    elif tech=='static':
        return size
    
def f(u): #change discretized function here
    return D1 @ u + D0 @ (u**2)
    #return D1 @ u + (D0 @ u)**2
        
def df(u): #change frichet derivative here
    return D1 + 2*D0*u
    #return D1 + 2*D0*u*D0
    
def phi(f):
    return .5*f.dot(f)
    
def dphi(f,df):
    return df.T @ f
    
    
def sobolev(u):
    gradH = lin.solve(A,u)
    return gradH
    
def graph(x,y1):
    plt.plot(x,y1)
    plt.show()
    
#---MAIN---
x = np.linspace(LBOUND,UBOUND,POINTS)
#y0 = np.zeros(POINTS)
y = np.random.rand(POINTS)
#yexact = np.exp(x)

if INITIAL is not None:
    index = np.argmin(abs(x-INITIAL[0]))
    y[index] = INITIAL[1]

while phi(f(y)) > EPSILON and np.isfinite(phi(f(y))):
<<<<<<< HEAD
    grad =  sobolev(dphi( f(y), df(y) ))
    s = step_size(y,grad,'static')
=======
    grad =  dphi( f(y), df(y) )
    if INITIAL is not None:
        grad[index] = 0
    s = step_size(f(y),f(grad),'dynamic')
>>>>>>> d7586146040a05c90e1c8ca1f578a2e5793b2b90
    y0 = np.copy(y)
    y -= s*grad
    if k%2 == 0:
        print(k, phi(f(y)))
        graph(x,y)
    k=k+1

graph(x,y)
print(k)
