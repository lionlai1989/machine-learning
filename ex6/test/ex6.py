#!/usr/bin/env python3

import sys
import os
import supportVectorMachine as svm
import scipy.io
from scipy.optimize import fmin_cg
import numpy as np
import matplotlib.pyplot as plt

# change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))
#print(os.getcwd())

########## Part 1: Loading and Visualizing Data ########## 
print('Loading and Visualizing Data')
data = scipy.io.loadmat("ex6data1.mat", mat_dtype=False)
#print(data)

x = data['X']
y = data['y']
print(x, y)
#print('x shape =', x.shape, '\ny shape =', y.shape)

pos = np.where(y==1)
neg = np.where(y==0)
print(type(pos), '\n', neg[0])
plt.scatter(x[pos[0], 0], x[pos[0], 1], marker='o', c='y')
plt.scatter(x[neg[0], 0], x[neg[0], 1], marker='x', c='k')

#input('Program paused. Press enter to continue...')

########## Part 2: Training Linear SVM ########## 
C = 1
model = svm.svmTrain(x, y, C, 'linear', 1e-3, 200, sigma=0)
svm.visualizeBoundaryLinear(x, y, model)
#svm.visualizeBoundaryLinear(x, y, model)
plt.show()
input('Program paused. Press enter to continue...')

########## Part 3: Implementing Gaussian Kernel ########## 
print('Evaluating the Gaussian Kernel ...\n')
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = svm.gaussianKernel(x1, x2, sigma)
print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = 0.5:\
        (this value should be 0.324652)\n', sim)

 
########## Part 4: Visualizing Dataset 2 ########## 
print('Loading and Visualizing Data')
data = scipy.io.loadmat("ex6data2.mat", mat_dtype=False)
#print(data)

x = data['X']
y = data['y']
#print(x, y)
#print('x shape =', x.shape, '\ny shape =', y.shape)

pos = np.where(y==1)
neg = np.where(y==0)
print(type(pos), '\n', neg[0])
plt.scatter(x[pos[0], 0], x[pos[0], 1], marker='o', c='y')
plt.scatter(x[neg[0], 0], x[neg[0], 1], marker='x', c='k')

########## Part 5: Training SVM with RBF Kernel (Dataset 2) ########## 
C = 1
model = svm.svmTrain(x, y, C, 'rbf', 1e-3, -1, sigma=0.1)
svm.visualizeBoundaryLinear(x, y, model)
plt.show()
input('Program paused. Press enter to continue...')

########## Part 6: Visualizing Dataset 3 ########## 
print('Loading and Visualizing Data')
data = scipy.io.loadmat("ex6data3.mat", mat_dtype=False)
#print(data)

x = data['X']
y = data['y']
#print(x, y)
#print('x shape =', x.shape, '\ny shape =', y.shape)
########## Part 7: Training SVM with RBF Kernel (Dataset 3) ########## 
data = scipy.io.loadmat("ex6data3.mat", mat_dtype=False)

x = data['X']
xVal = data['Xval']
y = data['y']
yVal = data['yval']
print(x, y, xVal, yVal)
print('x shape =', x.shape, '\ny shape =', y.shape)
print('xVal shape =', xVal.shape, '\nyVal shape =', yVal.shape)
C, sigma = svm.dataset3Params(x, y, xVal, yVal)

print(sys.version)

