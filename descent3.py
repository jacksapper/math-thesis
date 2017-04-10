import numpy as np
import matplotlib.pyplot as plt
import math

def error(v1,v2):	#taxicab
	return abs(v1[0]-v2[0])+abs(v1[1]-v2[1])

def df(x,y):
	return np.array([x, 120*y**3])

epsilon = 10**-7
step_size = 0.001

old_guess = np.array([-100.0,-100.0]) #arbitrarily far away from guess
guess = np.array([-1.,-1.])
x = [guess[0]]
y = [guess[1]]
i=0
while not(math.isnan(guess[1])) and error(old_guess,guess)>epsilon:
	old_guess = np.copy(guess)
	grad=df(old_guess[0],old_guess[1])
	step_size = np.dot(old_guess,grad) / np.dot(grad,grad)
	guess += step_size * -grad
	print ('Guess:',guess," Error:",error(old_guess,guess))
	x.append(guess[0])
	y.append(guess[1])
	i+=1
print ('Final approximation is ', guess)
print (error)
print ('i is ',i)
plt.plot(x,y)
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.show()