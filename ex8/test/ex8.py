#!/usr/bin/env python3

import sys
import os
import anomalyDetection as ad
import scipy.io
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt

# change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))
#print(os.getcwd())

########## Part 1: Load Example Dataset ########## 
print('Visualizing example dataset for outlier detection.')
data = scipy.io.loadmat("ex8data1.mat", mat_dtype=False)
#print(data)

x = data['X']
xVal = data['Xval']
yVal = data['yval']

print('x shape =', x.shape)
plt.plot(x[:,0], x[:,1], "bx")
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.axis([0, 30, 0, 30])

input('Program paused. Press enter to continue...')

########## Part 2: Estimate the dataset statistics ########## 
print('Visualizing Gaussian fit.')
mu, sigma2 = ad.estimateGaussian(x)
#print(mu.shape, sigma2.shape)
p = ad.multivariateGaussian(x, mu, sigma2)
ad.visualizeFit(x, mu, sigma2)
#plt.show()
input('Program paused. Press enter to continue...')

########## Part 3: Find Outliers ########## 
pVal = ad.multivariateGaussian(xVal, mu, sigma2)
epsilon, F1 = ad.selectThreshold(yVal, pVal)
print('Best epsilon found using cross-validation:', epsilon)
print('Best F1 on Cross Validation Set:', F1)
print('(you should see a value epsilon of about 8.99e-05)')
outliers = np.where(p < epsilon)
plt.plot( x[outliers, 0], x[outliers, 1], 'ro' \
	,markerfacecolor='none', markeredgecolor='r', linewidth=2, markersize=10 )
plt.show()
input('Program paused. Press enter to continue...')

########## Part 4: Multidimensional Outliers ########## 
data = scipy.io.loadmat("ex8data2.mat", mat_dtype=False)
#print(data)

x = data['X']
xVal = data['Xval']
yVal = data['yval']
print('x shape =', x.shape)

mu, sigma2 = ad.estimateGaussian(x)
p = ad.multivariateGaussian(x, mu, sigma2)
#p = ad.originalGaussian(x, mu, sigma2)
pVal = ad.multivariateGaussian(xVal, mu, sigma2)
#pVal = ad.originalGaussian(xVal, mu, sigma2)
epsilon, F1 = ad.selectThreshold(yVal, pVal)
print('Best epsilon found using cross-validation:', epsilon)
print('Best F1 on Cross Validation Set:, F1')
print('# Outliers found: ', np.sum(p<epsilon))
print('(you should see a value epsilon of about 1.38e-18)')

print(sys.version)