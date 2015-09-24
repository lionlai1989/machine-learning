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

def findClosestCentroids(x, initialCentroids):
    idx = np.zeros(0) # create (1, 2) array
    for x in x:
        a = (math.hypot(x[0]-initialCentroids[0][0], x[1]- initialCentroids[0][1]))
        b = (math.hypot(x[0]-initialCentroids[1][0], x[1]- initialCentroids[1][1]))
        c = (math.hypot(x[0]-initialCentroids[2][0], x[1]- initialCentroids[2][1]))
        if a < b and a < c:
            idx = np.append(idx, np.array([1]), axis=0)
        elif b < c:
            idx = np.append(idx, np.array([2]), axis=0)
        else: 
            idx = np.append(idx, np.array([3]), axis=0)
    return idx

def computeCentroids(x, idx, K):
    #print(x.shape, idx.shape)
    centroids = np.zeros(shape=(K, x.shape[1]))
    for i in range(1, K+1): # i = 1 to K
        #print(np.where(idx==i)[0])
        tmp = (np.where(idx==i)[0])
        centroids[i-1] = (np.array(np.mean(x[tmp], axis=0)))
    return centroids

def plotProgressKMeans(x, centroids, previousCentroids, idx, K, i):

    return 1

def runKMeans(x, initialCentroids, maxIters, flag):
    # initialize values
    m, n = x.shape
    K = initialCentroids.shape[0]
    print(m, n, K)
    centroids = initialCentroids
    previousCentroids = centroids
    
    # Run K-Means
    for i in range(1,maxIters):
        print('K-Means iteration', i, maxIters)
        idx = findClosestCentroids(x, centroids)

        if(flag):
            plotProgressKMeans(x, centroids, previousCentroids, idx, K, i)
            previousCentroids = centroids


        centroids = computeCentroids(x, idx, K)
    
    return centroids, idx

def kMeansInitCentroids(x, K):
    initCentroids = np.zeros(K, x.shape[1])
    return initCentroids


