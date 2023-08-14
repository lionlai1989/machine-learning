#!/usr/bin/env python3
import codecs
import sys
import os
import recommenderSystem as rs
import scipy.io
from scipy.misc import imread
from scipy.optimize import fmin_cg
import numpy as np
import matplotlib.pyplot as plt

# change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))
#print(os.getcwd())

########## Part 1: Loading movie ratings dataset ########## 
print('Loading movie ratings dataset.')
data = scipy.io.loadmat("ex8_movies.mat", mat_dtype=False)
#print(data)
y = data['Y']
#print(y[0,:]) # rating equals to 0~5
r = data['R']
print('Average rating for movie 1 (Toy Story):', np.mean(y[0,np.where(r[0,:]==1)]))
#fig, (ax1) = plt.subplots(nrows=1, figsize=(6, 10))
#ax1.imshow(y, extent=[0,942,0,1682], aspect='auto')
#plt.show()

########## Part 2: Collaborative Filtering Cost Function ########## 
data = scipy.io.loadmat("ex8_movieParams.mat", mat_dtype=False)
#print(data)
x = data['X']
theta = data['Theta']
numUsers = 4
numMovies = 5
numFeatures = 3
x = x[0:numMovies, 0:numFeatures]
theta = theta[0:numUsers, 0:numFeatures]
y = y[0:numMovies, 0:numUsers]
r = r[0:numMovies, 0:numUsers]
params = np.r_[x.flatten(), theta.flatten()]
#print(x.shape, theta.shape, y.shape ,r.shape, numUsers, numMovies, numFeatures)
J = rs.cofiCostFunc(params, y, r, numUsers, numMovies, numFeatures, 0)[0]
print('Cost at loaded parameters: ', J, \
	'(this value should be about 22.22)')

########## Part 3: Collaborative Filtering Gradient ########## 
#rs.checkCostFunction(0) not finished

########## Part 4: Collaborative Filtering Cost Regularization ########## 
J = rs.cofiCostFunc(params, y, r, numUsers, numMovies, numFeatures, 1.5)[0]
print('Cost at loaded parameters (lambda = 1.5): ', J, \
	'(this value should be about 31.34)')

########## Part 5: Collaborative Filtering Gradient Regularization ########## 
movieList = rs.loadMovieList()

########## Part 6: Entering ratings for a new user ########## 
# create a new user
myRatings = np.zeros((1682, 1))
myRatings[0] = 4
myRatings[97] = 2
myRatings[6] = 3
myRatings[11] = 5
myRatings[53] = 4
myRatings[63] = 5
myRatings[65] = 3
myRatings[68] = 5
myRatings[182] = 4
myRatings[225] = 5
myRatings[354] = 5
#tmp = np.zeros((1682, 1))
#tmp[np.where(myRatings != 0)] = 1

########## Part 7: Learning Movie Ratings ########## 
print('Training collaborative filtering...')
data = scipy.io.loadmat("ex8_movies.mat", mat_dtype=False)
y = data['Y']
r = data['R']
#y = np.append(myRatings, y, axis=1)
#r = np.append(myRatings>0, r, axis=1)
y = np.c_[myRatings, y]
r = np.c_[myRatings!=0, r]

yNorm, yMean = rs.normalizeRatings(y, r)
moviesNO, userNO = y.shape
featuresNO = 10
x = np.random.rand(moviesNO, featuresNO)
theta = np.random.rand(userNO, featuresNO)

lamda = 10
tmpCost = lambda params: rs.cofiCostFunc(params, y, r, userNO, moviesNO,\
        featuresNO, lamda)[0]
tmpGrad = lambda params: rs.cofiCostFunc(params, y, r, userNO, moviesNO,\
        featuresNO, lamda)[1]
params = np.r_[x.flatten(), theta.flatten()]
opt = fmin_cg(tmpCost, np.copy(params), fprime=tmpGrad, \
	gtol=1e-09, maxiter=100)

x = np.reshape(opt[0:moviesNO*featuresNO], (moviesNO, featuresNO))
theta = np.reshape(opt[moviesNO*featuresNO:], (userNO, featuresNO))
print('Recommender system learning completed.')

########## Part 8: Recommendation for you ########## 
predict = np.dot(x, theta.T)
myPredict = predict[:, 0]+yMean
ix = np.argsort(myPredict) # sort myPredict in ascending order and return index

#for k in range(1682):
#	print(ix[k])
#	print(myPredict[ix[k]])
print('Top recommendations for you:')
for i in range(1,11):
	j = ix[-i]
	print('Predicting rating', myPredict[j], 'for movie', movieList[j])

print(sys.version)