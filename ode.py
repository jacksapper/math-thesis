# -*- coding: utf-8 -*-
#---IMPORTS---
import numpy as np
import matplotlib.pyplot as plt
#---CONSTANTS---
LBOUND = 0.
UBOUND = 1.
POINTS = 2**7
EPSILON = 10**-9
INITIAL = (0,1)

#Matrix is O(POINTS**2)

#---DERIVED CONSTANTS---DON'T CHANGE THESE, CHANGE THE REAL CONSTANTS
INTERVAL_LENGTH = (UBOUND-LBOUND)/(POINTS-1)
D0 = .5*(np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
D1 = (1/INTERVAL_LENGTH)*(-1*np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1))
D = D1-D0
A = D1.T @ D1 + D0.T @ D0
k = 1

display = [1,2,3,4,5,6]

#---FUNCTIONS---        
def step_size(u, v, tech='dynamic', size=EPSILON/10):
    if tech=='dynamic':
        upper = u.dot(v)
        lower = v.dot(v)
        return upper/lower
    elif tech=='static':
        return size
    
def f(u):
    f = D @ u
    result = (f).dot(f)
    return .5*result
    
#u and v need to be column matrices not arrays for the * operator to work correctly
def df(u):
    grad2=(D.T @ D) @ u
    if INITIAL is not None:
        grad2[index] = 0
    return grad2
    
def sobolev(u):
    gradH = np.linalg.solve(A,u)
    if INITIAL is not None:
        gradH[index] = 0
    return gradH
    
def graph(x,y1,y2):
    plt.plot(x,y1, 'r--',label='Approximation')
    plt.plot(x,y2, label='Exact Solution')
    plt.legend(loc='lower left')
    #plt.savefig('/home/jason/Dropbox/thesis/img/good-triv-bdd/{num}.png'.format(num=k), dpi=150)
    #plt.show()
    
#---MAIN---
x = np.linspace(LBOUND,UBOUND,POINTS).T
yold = np.zeros(POINTS).T
ynew = 2. * np.ones(POINTS).T
yexact = np.exp(x)

if INITIAL is not None:
    index = np.argmin(abs(x-INITIAL[0]))
    ynew[index] = INITIAL[1]

while f(ynew) > EPSILON and k <= max(display):
    grad = sobolev(df(ynew))
    s = step_size((D @ ynew),(D @ grad),'dynamic')
    #s = 10**-5
    yold = np.copy(ynew)
    ynew = yold - s*grad
    if k in display:
        print("|",k,"|", round(f(ynew),3), "|", round(s,3),"|", round(ynew[POINTS//2],3),"|",round(grad[POINTS//2],3),"|")
        plt.plot(x,ynew,label='k={num}'.format(num=k))
    k=k+1

#graph(x,ynew,yexact)
#print("|",k,"|", round(f(ynew),3), "|", round(s,3),"|", round(ynew[POINTS//2],3),"|",round(grad[POINTS//2],3),"|")
plt.legend(loc='lower right')
plt.savefig('/home/jason/Dropbox/thesis/newimg/sob-dyn-bdd.png',dpi=150)
plt.show()