#!/usr/bin/env python3

import sys
import os
import linearRegBiasVariance as lr
import scipy.io
from scipy.optimize import fmin_cg
import numpy as np
import matplotlib.pyplot as plt

# change python current working directory to the current folder
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

plt.plot(x, y, 'ro')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

input('Program paused. Press enter to continue...')

########## Part 2: Regularized Linear Regression Cost ########## 
tmpX = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
theta = np.array([[1], [1]])
# python is a weird language, when you want to ONLY pass VALUE of an array, 
# DO NOT pass "theta", pass "np.copy(theta)", otherwise the theta itself will
# be changed if you change theta in the function.
cost = lr.linearRegCostFunction(tmpX, y, np.copy(theta), lamda=1)[0]
print('Cost at theta = (1, 1) should be 303.993192\nthe calculated cost is', \
        cost)

input('Program paused. Press enter to continue...')

########## Part 3: Regularized Linear Regression Gradient ########## 
grad = lr.linearRegCostFunction(tmpX, y, np.copy(theta), lamda=1)[1]
print('Gradient at theta = (1, 1) should be (-15.303016, 598.250744)\n\
the calculated grad is\n', grad)

input('Program paused. Press enter to continue...')

########## Part 4: Train Linear Regression ########## 
theta = lr.trainLinearReg(tmpX, y, lamda=0)
plt.plot(x, x*(theta[1])+theta[0])
plt.show()

########## Part 5: Learning Curve for Linear Regression ########## 
tmpXv = np.concatenate((np.ones((xv.shape[0], 1)), xv), axis=1)
errTrain, errVal = lr.learningCurve(tmpX, y, tmpXv, yv, lamda=0)
u, v = errTrain.shape
tmp = np.array(range(u+1))
plt.plot(tmp[1:], errTrain, label='Train',)
plt.plot(tmp[1:], errVal, label='Cross Validation')
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.title('Learning curve for linear regression')
plt.show()

input('Program paused. Press enter to continue...')

########## Part 6: Feature Mapping for Polynomial Regression ########## 
# Map x onto polynomial features and normalize
p=8
lamda = 3
polyX = lr.polyFeatures(np.copy(x), p)
polyX, mean, std = lr.featureNormalize(np.copy(polyX))
polyX = np.concatenate((np.ones((polyX.shape[0], 1)), polyX), axis=1)

# Map xt onto polynomial features and normalize(using mean and std)
polyXt = lr.polyFeatures(np.copy(xt), p)
polyXt = polyXt - mean
polyXt = polyXt / std
polyXt = np.concatenate((np.ones((polyXt.shape[0], 1)), np.copy(polyXt)), axis=1)

# Map xv onto polynomial features and normalize(using mean and std)
polyXv = lr.polyFeatures(np.copy(xv), p)
polyXv = polyXv - mean
polyXv = polyXv / std
polyXv = np.concatenate((np.ones((polyXv.shape[0], 1)), np.copy(polyXv)), axis=1)

print('First Row of Normalized Training Example is', polyX[0,:])

input('Program paused. Press enter to continue...')

########## Part 7: Learning Curve for Polynomial Regression ########## 

theta = lr.trainLinearReg(polyX, y, lamda)
plt.plot(x, y, 'ro')
lr.plotFit(min(x), max(x), mean, std, theta, p)
plt.title('Polynomial Regression Fit (lambda = %s)'%lamda)
plt.show()

errTrain, errVal = lr.learningCurve(polyX, y, polyXt, yt, lamda)
u, v = errTrain.shape
tmp = np.array(range(u+1))
plt.plot(tmp[1:], errTrain, label='Train',)
plt.plot(tmp[1:], errVal, label='Cross Validation')
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.title('Polynomial Regression Learning Curve (lambda = %s)'%lamda)
plt.show()

input('Program paused. Press enter to continue...')

########## Part 8: Validation for Selecting Lamda ########## 
lamdaVec, errTrain, errVal = lr.validationCurve(polyX, y, polyXt, yt)
plt.plot(lamdaVec[0:], errTrain, label='Train')
plt.plot(lamdaVec[0:], errVal, label='Cross Validation')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()

print(sys.version)

