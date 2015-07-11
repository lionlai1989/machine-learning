#!/usr/bin/env python3

import sys
import os
import linearRegBiasVariance as lr
import scipy.io
from scipy.optimize import fmin_cg
import numpy as np
import matplotlib.pyplot as plt

#change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))
#print(os.getcwd())

########## Part 1: Loading and Visualizing Data ########## 
print('Loading and Visualizing Data')
data = scipy.io.loadmat("ex5data1.mat", mat_dtype=False)
x = data['X']
xv = data['Xval']
xt = data['Xtest']
y = data['y']
yv = data['yval']
yt = data['ytest']
#print(x, y, xv, yv)
print('x shape =', x.shape, '\ny shape =', y.shape)
print('xv shape =', xv.shape, '\nyv shape =', yv.shape)
print('xt shape =', xt.shape, '\nyt shape =', yt.shape)
'''
plt.plot(x, y, 'ro')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()
input('Program paused. Press enter to continue...')
'''
########## Part 2: Regularized Linear Regression Cost ########## 
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
theta = np.array([[1], [1]])
# python is a weird language, when you want to ONLY pass VALUE of an array, 
# DO NOT pass "theta", pass "np.copy(theta)", otherwise the theta itself will
# be changed.
cost = lr.linearRegCostFunction(x, y, np.copy(theta), lamda=1)[0]
print('Cost at theta = (1, 1) should be 303.993192\nthe calculated cost is', cost)

########## Part 3: Regularized Linear Regression Gradient ########## 
grad = lr.linearRegCostFunction(x, y, np.copy(theta), lamda=1)[1]
print('Gradient at theta = (1, 1) should be (-15.303016, 598.250744)\n\
the calculated grad is\n', grad)

########## Part 4: Train Linear Regression ########## 
theta = lr.trainLinearReg(x, y, lamda=1)
print(theta, theta.shape)

########## Part 5: Learning Curve for Linear Regression ########## 
xv = np.concatenate((np.ones((xv.shape[0], 1)), xv), axis=1)
lr.learningCurve(x, y, xv, yv, lamda=0)

########## Part 6: Feature Mapping for Polynomial Regression ########## 
lr.polyFeatures(x, p=2)

print(sys.version)

