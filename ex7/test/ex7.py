#!/usr/bin/env python3

import sys
import os
import kMeansClustering as kmc
import scipy.io
from scipy.optimize import fmin_cg
import numpy as np
import matplotlib.pyplot as plt

# change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))
#print(os.getcwd())

########## Part 1: Find Closest Centroids ########## 
print('Finding closest centroids')
data = scipy.io.loadmat("ex7data2.mat", mat_dtype=False)
print(data)

x = data['X']
print('x shape =', x.shape)

K = 3
initialCentroids = np.array([[3, 3], [6, 2], [8, 5]])

idx = kmc.findClosestCentroids(x, initialCentroids)

print('Closest centroids for the first 3 examples:')
print(idx[0:3])
print('the closest centroids should be 1, 3, 2 respectively')

########## Part 2: Compute Means ########## 
print('Computing centroids means.')
centroids = kmc.computeCentroids(x, idx, K)
print(centroids)
########## Part 3: K-Means Clustering ########## 

print('Running K-Means clustering on example dataset.')
data = scipy.io.loadmat("ex7data2.mat", mat_dtype=False)
K = 3
maxIters = 3
initialCentroids = np.array([[3, 3], [6, 2], [8, 5]])
centroids, idx = kmc.runKMeans(x, initialCentroids, maxIters, True)
print(centroids)

########## Part 4: K-Means Clustering on Pixels ########## 
'''
y = data['y']
#print(x, y)
#print('x shape =', x.shape, '\ny shape =', y.shape)

pos = np.where(y==1)
neg = np.where(y==0)
print(type(pos), '\n', neg[0])
plt.scatter(x[pos[0], 0], x[pos[0], 1], marker='o', c='y')
plt.scatter(x[neg[0], 0], x[neg[0], 1], marker='x', c='k')

########## Part 5: Image Compression ########## 
input('Program paused. Press enter to continue...')
'''
print(sys.version)

