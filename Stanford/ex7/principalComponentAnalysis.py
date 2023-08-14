#!/usr/bin/env python3

import math
import sys
import numpy as np
import scipy.io 
from types import *
import random

from scipy.optimize import fmin_cg

import matplotlib.pyplot as plt

from plotly.plotly import *
from plotly.graph_objs import *
from sklearn import svm

def featureNormalize(x):
	mean = np.mean(x, axis=0)
	tmp = (x - mean)
	# ddof is Delta Degrees of Freedom,
	# the divisor used in calculations is N - ddof
	std = np.std(tmp, axis=0, ddof=0)
	normX = tmp/std
	return normX, mean, std

def pca(normX):
	# Singular Value Decomposition
	m, n = normX.shape
	covariance = (np.dot(normX.T, normX))/m
	u, s, v = np.linalg.svd(covariance) # u is eigenVector, s is eigenValue
	'''
	print(u.shape, s.shape, v.shape) # u = n*n, s = n, v = n*n
	print(u)
	print(s)
	print(v)
	'''
	return u, s

def projectData(normX, eigU, K):
	uReduced = eigU[:, :K]
	return np.dot(normX, uReduced)

def recoverData(z, eigU, K):
	uReduced = eigU[:, :K]
	return np.dot(z, uReduced.T)

def displayData(x, num):
	'''
	It randomly pick num data in x and displayed 2D data in a nice grid.
	'''
	print("Visualize", num, "selected data...")
	# return num random training example
    #idxs = np.random.randint(x.shape[0], size=num)	
	idxs = range(num)
	tmp = np.sqrt(num)
	num = tmp.astype(np.int64)
	fig, ax = plt.subplots(num, num)
	img_size = math.sqrt(x.shape[1])
	for i in range(num):
		for j in range(num):
            # the array of image is colummn-by-column indexing
			xi = x[idxs[i * num + j], :].reshape(img_size, img_size).T 
			ax[i, j].set_axis_off()
			ax[i, j].imshow(xi, aspect="auto", cmap="gray")
	plt.show()
	return ax
	
