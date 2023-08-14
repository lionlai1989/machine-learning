#!/usr/bin/env python3

import sys
import os
import neuralNetworksLearning as nn
import math
import numpy as np
import scipy.io 
from types import *
import random

from scipy.optimize import fmin_cg

import matplotlib.pyplot as plt

from plotly.plotly import *
from plotly.graph_objs import *

#change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))
#print(os.getcwd())

########## Part 1: Loading and Visualizing Data ########## 
data = scipy.io.loadmat("ex4data1.mat", mat_dtype=False)
x = data['X']
y = data['y']
y[y==10] = 0 # '0' is encoded as '10' in data, fix it, y = 0 1 2 3 4 5 6 7 8 9
print('x shape =', x.shape, '\ny shape =', y.shape)

data = scipy.io.loadmat("ex4weights.mat", mat_dtype=False)
theta1 = data['Theta1']
theta2 = data['Theta2']
print('theta1 shape =', theta1.shape, '\ntheta2 shape =', theta2.shape)

nn.displayData(x, 100)

########## Part 2: Preparing theta and input x ########## 
theta = np.concatenate((theta1.flatten('C'), theta2.flatten('C')))
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

########## Part 3: Compute Cost (Feedforward) ########## 
J = nn.nnCostFunction(theta, x.shape[1]-1, theta1.shape[0], theta2.shape[0], x, y, lamda=0)[0]
print("Cost of unregularized feedforward function(lambda=0) =", J)

########## Part 4: Implement Regularization ########## 
J = nn.nnCostFunction(theta, x.shape[1]-1, theta1.shape[0], theta2.shape[0], x, y, lamda=1)[0]
print("Cost of regularized feedforward function(lambda=1) =", J)

########## Part 5: Sigmoid Gradient ########## 
print('Sigmoid gradient evaluated at [1, -0.5, 0, 0.5, 1] =', nn.sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1])))

########## Part 6: Implement Backpropagation ########## 
print('running checkNNGradients to check backpropagation...')
diff = nn.checkNNGradients(lamda=3)
print('If the backpropagation is correct, then the relative difference will be less than 1e-9.\nRelative Difference is', diff)
debugJ = nn.nnCostFunction(theta, x.shape[1]-1, theta1.shape[0], theta2.shape[0], x, y, lamda=3)[0]
print('Cost at (fixed) debugging parameters, Debug cost should be 0.576051.\nDebug cost is', debugJ)

########## Part 7: Training Neural Network ########## 
print('training NN network...\n')
initTheta1, initTheta2 = nn.randInitializeWeight(theta1, theta2, epislon=0.12)
initTheta = np.concatenate((initTheta1.flatten('C'), initTheta2.flatten('C')))
xopt = fmin_cg(nn.computeCost, initTheta, fprime=nn.computeGradient, 
	args=(x.shape[1]-1, theta1.shape[0], theta2.shape[0], x, y, 1), gtol=1e-05, maxiter=50)
nntheta1 = np.reshape(xopt[:(x.shape[1]-1+1)*theta1.shape[0]], (theta1.shape[0], x.shape[1]-1+1))
nntheta2 = np.reshape(xopt[(x.shape[1]-1+1)*theta1.shape[0]:], (theta2.shape[0], theta1.shape[0]+1))

########## Part 8: Visualize Weights ########## 
nn.displayData(nntheta1[:,1:], 25)

########## Part 9: Implement Predict ########## 
result = nn.predict(nntheta1, nntheta2, x, y)
print('Neural Network training accuracy is', np.mean(result)*100)

print(sys.version)
