import numpy as np
import matplotlib.pyplot as plt
import math

def error(v1,v2):	#taxicab
	return abs(v1[0]-v2[0])+abs(v1[1]-v2[1])

def df(x,y):
	return np.array([3*x**2 - 4*x, 3*y**2 + 6*y])

epsilon = 10**-7
step_size = 0.1

old_guess = np.array([-100.0,-100.0]) #arbitrarily far away from guess
guess = np.array([0.5, 0.5])
x = [guess[0]]
y = [guess[1]]
while not(math.isnan(guess[1])) and error(old_guess,guess)>epsilon:
	old_guess = np.copy(guess)
	guess += step_size * -df(old_guess[0],old_guess[1])
	print 'Guess:',guess," Error:",error(old_guess,guess)
	x.append(guess[0])
	y.append(guess[1])
print'Final approximation is ', guess
print error

plt.plot(x,y)
plt.show()
