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
    u, v = x.shape
    m, n = initialCentroids.shape
    #print(u, v, m, n)
    idx = np.zeros(0)
    for x in x:
        tmp = np.zeros(0)
        for i in initialCentroids:
            tmp = np.append(tmp, np.array([np.linalg.norm(x-i)]), axis=0)
        idx = np.append(idx, np.array([1 + np.argmin(tmp)]), axis=0)  
    return idx

def computeCentroids(x, idx, K):
    centroids = np.zeros(shape=(K, x.shape[1]))
    for i in range(1, K+1): # i = 1 to K
        tmp = (np.where(idx==i)[0])
        centroids[i-1] = (np.array(np.mean(x[tmp], axis=0)))
    return centroids

def plotProgressKMeans(x, centroids, previousCentroids, idx, K, i):
    '''
    I don't impliment the block of code, since I tend to leave some blank here.
    '''
    return 1

def runKMeans(x, initialCentroids, maxIters, flag):
    # initialize values
    m, n = x.shape
    K = initialCentroids.shape[0]
    #print(m, n, K)
    centroids = initialCentroids
    previousCentroids = centroids
    # Run K-Means
    for i in range(1,maxIters+1):
        print('K-Means iteration', i, 'of', maxIters)
        idx = findClosestCentroids(x, centroids)
        if(flag):
            plotProgressKMeans(x, centroids, previousCentroids, idx, K, i)
            previousCentroids = centroids
        centroids = computeCentroids(x, idx, K)
        print(centroids)
    return centroids, idx

def kMeansInitCentroids(x, K):
    np.random.shuffle(np.copy(x)) 
    return x[:K,:]


