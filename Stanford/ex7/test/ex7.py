#!/usr/bin/env python3

import sys
import os
import kMeansClustering as kmc
import scipy.io
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt

# change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))
#print(os.getcwd())

########## Part 1: Find Closest Centroids ########## 
print('Finding closest centroids')
data = scipy.io.loadmat("ex7data2.mat", mat_dtype=False)
#print(data)

x = data['X']
print('x shape =', x.shape)

K = 3
initialCentroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = kmc.findClosestCentroids(x, initialCentroids)
print('Closest centroids for the first 3 examples:')
print(idx[0:3])
print('the closest centroids should be 1, 3, 2 respectively')
input('Program paused. Press enter to continue...')

########## Part 2: Compute Means ########## 
print('Computing centroids means.')
centroids = kmc.computeCentroids(x, idx, K)
print('Centroids computed after initial finding of closest centroids:')
print(centroids)
print('the centroids should be')
print('[ 2.428301 3.157924 ]')
print('[ 5.813503 2.633656 ]')
print('[ 7.119387 3.616684 ]')
input('Program paused. Press enter to continue...')

########## Part 3: K-Means Clustering ########## 
print('Running K-Means clustering on example dataset.')
data = scipy.io.loadmat("ex7data2.mat", mat_dtype=False)
K = 3
maxIters = 10
#initialCentroids = np.array([[3, 3], [6, 2], [8, 5]])
initialCentroids = kmc.kMeansInitCentroids(x, K) 
# if picking random centroids from example data, maxIters should be increased.
centroids, idx = kmc.runKMeans(x, initialCentroids, maxIters, False)
print('final centroids:\n', centroids)
print('K-Means Done')
input('Program paused. Press enter to continue...')

########## Part 4: K-Means Clustering on Pixels ########## 
print('Running K-Means clustering on pixels from an image')
pic = imread('me.png')

axes = plt.gca()
figure = plt.gcf()
axes.imshow(pic)
plt.show(block=True)

pic = pic / 255
a = pic.shape[0]
b = pic.shape[1]
c = pic.shape[2]
# reshape pic from a*b*c to (a*b)*c
pic = np.reshape(pic, (a*b, c))
K = 16
maxIters = 10
# again, the np.copy(x) here is very important, if not, x will be changed.
initialCentroids = kmc.kMeansInitCentroids(np.copy(pic), K)
centroids, idx = kmc.runKMeans(np.copy(pic), initialCentroids, maxIters, False)

########## Part 5: Image Compression ########## 
print('Applying K-Means to compress an image.')
idx = kmc.findClosestCentroids(pic, np.copy(centroids))
xRecovered = np.zeros(pic.shape)
for i in range(0, idx.shape[0]):
	tmp = int(idx[i])
	xRecovered[i] = centroids[tmp-1]
xRecovered = np.reshape(xRecovered, (a, b, c))

axes = plt.gca()
figure = plt.gcf()
axes.imshow( xRecovered)
plt.show( block=True )

print(sys.version)