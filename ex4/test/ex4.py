#!/usr/bin/python3
#shebang is good

import sys
import os
import neuralNetworksLearning as nn
import math
import numpy as np
import scipy.io 
from types import *
from mpl_toolkits.mplot3d import Axes3D
import random

from scipy.optimize import fmin_cg

import matplotlib.pyplot as plt
import matplotlib as mpl

from plotly.plotly import *
from plotly.graph_objs import *

#change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))
#print(os.getcwd())

data = scipy.io.loadmat("ex4data1.mat", mat_dtype=False)
x = data['X']
y = data['y']
y[y==10] = 0 # '0' is encoded as '10' in data, fix it, y = 0 1 2 3 4 5 6 7 8 9
print('x shape =', x.shape, '\ny shape =', y.shape)

data1 = scipy.io.loadmat("ex4weights.mat", mat_dtype=False)
theta1 = data1['Theta1']
theta2 = data1['Theta2']
print('theta1 shape =', theta1.shape, '\ntheta2 shape =', theta2.shape)

nn.displayData(x, 100)

theta = np.concatenate((theta1.flatten('C'), theta2.flatten('C')))
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

J = nn.nnCostFunction(theta, x.shape[1]-1, theta1.shape[0], theta2.shape[0], x, y, lamda=0)[0]
print("Cost of unregularized feedforward function(lambda=0) =", J)
J = nn.nnCostFunction(theta, x.shape[1]-1, theta1.shape[0], theta2.shape[0], x, y, lamda=1)[0]
print("Cost of regularized feedforward function(lambda=1) =", J)

print('Sigmoid gradient evaluated at [1, -0.5, 0, 0.5, 1] =', nn.sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1])))

print('running checkNNGradients to check backpropagation...')
nn.checkNNGradients(lamda=3)
debugJ = nn.nnCostFunction(theta, x.shape[1]-1, theta1.shape[0], theta2.shape[0], x, y, lamda=3)[0]
print('Debug cost is', debugJ)

print('training NN network...\n')
initTheta1, initTheta2 = nn.randInitializeWeight(theta1, theta2, epislon=0.12)
initTheta = np.concatenate((initTheta1.flatten('C'), initTheta2.flatten('C')))
xopt = fmin_cg(nn.computeCost, initTheta, fprime=nn.computeGradient, 
	args=(x.shape[1]-1, theta1.shape[0], theta2.shape[0], x, y, 1), gtol=1e-05, maxiter=50)
nntheta1 = np.reshape(xopt[:(x.shape[1]-1+1)*theta1.shape[0]], (theta1.shape[0], x.shape[1]-1+1))
nntheta2 = np.reshape(xopt[(x.shape[1]-1+1)*theta1.shape[0]:], (theta2.shape[0], theta1.shape[0]+1))
nn.displayData(nntheta1[:,1:], 25)
result = nn.predict(nntheta1, nntheta2, x, y)
print('Neural Network training accuracy is', np.mean(result)*100)

print(sys.version)
