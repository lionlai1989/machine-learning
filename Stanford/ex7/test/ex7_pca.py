#!/usr/bin/env python3

import sys
import os
import principalComponentAnalysis as pca
import kMeansClustering as kmc
from scipy.misc import imread
import scipy.io
from scipy.optimize import fmin_cg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))
#print(os.getcwd())

########## Part 1: Load Example Dataset ########## 
print('Visualizing example dataset for PCA.')
data = scipy.io.loadmat("ex7data1.mat", mat_dtype=False)
x = data['X']

plt.plot(x[:,0], x[:,1], "bo")
plt.axis([0.5, 6.5, 2.0, 8.0])

########## Part 2: Principal Component Analysis ########## 
print('shape of data:', x.shape)
print('Running PCA on example dataset')
normX, mean, std = pca.featureNormalize(x)
eigU, eigS = pca.pca(np.copy(normX)) # u is eigenVector, s is eigenValue
mu_1 = mean + 1.5 * eigS[0] * eigU[:, 0] # this is vector product, 
mu_2 = mean + 1.5 * eigS[1] * eigU[:, 1] # using 1.5 because of *.m.
# The fist line and the second line may not look perpendicular, 
# the reason is axis scaling.
plt.plot((mean[0], mean[0]+eigS[0]*1.5*eigU[0, 0]), \
	(mean[1], mean[1]+eigS[0]*1.5*eigU[1, 0]), 'k-', lw=2)
plt.plot((mean[0], mean[0]+eigS[1]*1.5*eigU[1, 0]), \
	(mean[1], mean[1]+eigS[1]*1.5*eigU[1, 1]), 'k-', lw=2)
plt.show()
print('Top eigenvector: ')
print(' u[:, 0] =', eigU[:, 0])
print('(you should expect to see -0.707107 -0.707107)')
input('Program paused. Press enter to continue...')

########## Part 3: Dimension Reduction ########## 
print('Dimension reduction on example dataset.')
print('Plotting normalized data')
plt.plot(normX[:,0], normX[:,1], "bo")
plt.axis([-4, 3, -4, 3])
plt.show()
K = 1;
z = pca.projectData(np.copy(normX), eigU, K)
#print(z.shape)
print('Projection of the first example:', z[0, :])
print('(this value should be about 1.481274)')
xRecover  = pca.recoverData(z, eigU, K);
#print(xRecover.shape)
print('Approximation of the first example:', xRecover[0, :])
print('(this value should be about  -1.047419 -1.047419)')

print('Draw lines connecting the projected points to the original points')
plt.plot(xRecover[:,0], xRecover[:,1], "ro")
plt.plot(normX[:,0], normX[:,1], "bo")

for i in range(xRecover.shape[0]): # i = 0 to 49
	plt.plot((xRecover[:,0], normX[:,0]), (xRecover[:,1], normX[:,1]), \
		'k-', lw=1, ls='--')
plt.axis([-4, 3, -4, 3])
plt.show()
input('Program paused. Press enter to continue...')

########## Part 4: Loading and Visualizing Face Data ########## 
print('Loading face dataset')
face = scipy.io.loadmat("ex7faces.mat", mat_dtype=False)
x = face['X']
print('shape of face:', x.shape)
pca.displayData(x[:100,:], 100);
input('Program paused. Press enter to continue...')

########## Part 5: PCA on Face Data: Eigenfaces ########## 
print('Running PCA on face dataset')
print('(this mght take a minute or two ...)')
normX, mean, std = pca.featureNormalize(x)
eigU, eigS = pca.pca(np.copy(normX))
pca.displayData(eigU[:,:36].T, 36);
input('Program paused. Press enter to continue...')

########## Part 6: Dimension Reduction for Faces ########## 
print('Dimension reduction for face dataset')
K = 100;
z = pca.projectData(normX, eigU, K);
print('The projected data Z has a size of:', z.shape)
input('Program paused. Press enter to continue...')

########## Part 7: Visualization of Faces after PCA Dimension Reduction ##########
print('Visualizing the projected (reduced dimension) faces')
K = 100;
xRecover  = pca.recoverData(z, eigU, K);
print('print Original faces')
pca.displayData(normX[:100, :], 100);
print('print Recovered faces')
pca.displayData(xRecover[:100, :], 100);
input('Program paused. Press enter to continue...')

########## Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ##########
print('Visualizing the 3-D result of k-Means')
A = imread('bird_small.png')
A = A / 255.0
img_size = A.shape
X = A.reshape( img_size[0] * img_size[1], img_size[2] )
K = 16
max_iters = 10
initial_centroids = kmc.kMeansInitCentroids( np.copy(X), K )
centroids, idx = kmc.runKMeans( np.copy(X), initial_centroids, max_iters, False)

fig = plt.figure()
ax = fig.add_subplot( 111, projection='3d' )
ax.scatter( X[:1000, 0], X[:1000, 1], X[:1000, 2], c=idx[:1000], marker='o' )
ax.set_xlim3d(0, 1)
ax.set_ylim3d(0, 1)
ax.set_zlim3d(0, 1)
ax.set_xlabel('R value')
ax.set_ylabel('G vlaue')
ax.set_zlabel('B value')
plt.show(block=True)
print('Pixel dataset plotted in 3D. Color shows centroid memberships')
print('The result is different from the ex7.pdf,')
print('because ex7_pca.m randomly picks 1000 data points')
input('Program paused. Press enter to continue...')

########## Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ########## 
X_norm, mu, sigma = pca.featureNormalize( np.copy(X) )
U, S = pca.pca( np.copy(X_norm) )
Z = pca.projectData( np.copy(X_norm), U, 2 ) # 3-D to 2-D
plt.scatter( Z[:1000, 0], Z[:1000, 1], c=idx[:1000], marker='o' )
plt.show(block=True)
print('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
print('The result is different from the ex7.pdf,')
print('because ex7_pca.m randomly picks 1000 data points')
print(sys.version)
