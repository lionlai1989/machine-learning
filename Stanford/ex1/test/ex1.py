#!/usr/bin/env python3
#shebang is good
import os
import sys
import linearRegression as lr
import numpy as np
import matplotlib.pyplot as plt
from plotly.plotly import *
from plotly.graph_objs import *
from types import *
from mpl_toolkits.mplot3d import Axes3D
import random

os.chdir(os.path.dirname(sys.argv[0]))

#Linear regression with one varaible
#Warm up exercise
print("Warm up erercise\n", np.identity(5), "\n")
input("press ENTER to continue...\n")

data = np.loadtxt("ex1data1.txt", delimiter = ',', unpack = False)
data = np.matrix(data) # cast 1-D to 2-D array
x = data[:, 0] # x is input vector, a 2-D array
y = data[:, 1] # y is real price of the house, a 2-D array
plt.plot(x, y, "ro")
plt.xlabel("Profit in $10,000s")
plt.ylabel("Population of City in 10,000s")

#initialize theta as 0
theta = np.zeros((x.shape[1] + 1, y.shape[1])) 

g = lr.linearRegression(x, y, iterations=1500)

#FIRST computeCost
print("First computeCost of ex1data1.txt: ", g.computeCost(x, y, theta))

#computeCost and gradientDescent for one variable
theta = g.batchGradientDescent(x, y, theta, rate=0.01)
print("After ", g.iterations, "iteration the result showed below")
print("Last computeCost of ex1data1.txt: ", g.computeCost(x, y, theta))
print("theta0 = ", theta[0], "\ntheta1 = ", theta[1])
print("predict1 = [1, 3.5] * theta = ", np.dot([1, 3.5], theta))
print("predict2 = [1, 7] * theta = ", np.dot([1, 7], theta))
plt.plot(x, x*(theta[1])+theta[0])

print("visualizing J(theta)...")
input("press ENTER to continue...")
#visualizing J(theta)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
theta0 = np.arange(-10.0, 10.0, 0.05)
theta1 = np.arange(-1.0, 4.0, 0.05)
T0, T1 = np.meshgrid(theta0, theta1)
zs = np.array([g.draw3DCost(x, y, theta0, theta1) for theta0,theta1 in zip(np.ravel(T0), np.ravel(T1))])
J = zs.reshape(T0.shape)

ax.plot_surface(T0, T1, J)

ax.set_xlabel("theta0")
ax.set_ylabel("theta1")
ax.set_zlabel("J function")

plt.show()
print("opening ex1data2.txt...")
input("press ENTER to continue...")

data = np.loadtxt("ex1data2.txt", delimiter = ',', unpack = False)
data = np.matrix(data)
x = data[:, 0:2] # x is input vector, a 2-D array
y = data[:, 2] # y is real price of the house, a 2-D array
#initialize theta as 0

g = lr.linearRegression(x, y, iterations=400)
n_x, mean, std = g.featureNormalize(x)
theta = np.zeros((n_x.shape[1] + 1, y.shape[1])) 

theta = g.batchGradientDescent(n_x, y, theta, rate=0.8)
print("size = 1650, bed rooms NO. = 3,")
tmp = ([1650, 3]-mean)/std
tmp = np.concatenate(([[1]], tmp), axis=1)
print( "linearRegression predict money is:", np.dot(tmp, theta) )
print( "normalEqn predict money is: ", np.dot([1, 1650, 3],(g.normalEqn(x, y))) )

print(sys.version)

