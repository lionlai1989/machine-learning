#!/usr/bin/env python3
import codecs
import io
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

def cofiCostFunc(params, y, r, numUsers, numMovies, numFeatures, lamda):      
    #print(type((np.where(r == 1))))
    #print(params.shape)
    x = np.reshape(params[0:numMovies*numFeatures], (numMovies, numFeatures))
    theta = np.reshape(params[numMovies*numFeatures:], (numUsers, numFeatures))
    #print(x.shape, theta.shape)
    tmp =  np.dot(x, np.transpose(theta)) - y
    tmp[np.where(r != 1)] = 0
    J = 0.5 * sum(sum(np.multiply(tmp, tmp))) + \
        0.5 * lamda*sum(sum(np.multiply(x, x))) + \
        0.5 * lamda*sum(sum(np.multiply(theta, theta)))
    xGrad = np.dot(tmp, theta)+0.5*lamda*x
    thetaGrad = np.dot(tmp.T, x)+0.5*lamda*theta
    grad = np.r_[xGrad.flatten(), theta.flatten()]
    return J, grad
'''
def computeNumericalGradient(J, x, theta):

    return 1


def checkCostFunction(lamda):
    tmpX = np.random.rand(4, 3)
    tmpTheta = np.random.rand(5, 3)
    y = np.dot(tmpX, tmpTheta.T)
    r = np.copy(y)
    #y[np.where( np.random.rand(y.shape[0], y.shape[1])>0.5)] = 0
    #y[np.where(y!=0)] = 1
    r[np.where( np.random.rand(r.shape[0], r.shape[1])>0.5)] = 0
    r[np.where(r!=0)] = 1
    print(y)
    x = np.random.rand(tmpX.shape[0], tmpX.shape[1]);
    theta = np.random.rand(tmpTheta.shape[0], tmpTheta.shape[1]);
    userNO = y.shape[1]
    moviesNO = y.shape[0]
    featuresNO = tmpTheta.shape[1]
    tmpCofi = lambda x, theta: cofiCostFunc((x), (theta), \
        y, r, userNO, moviesNO, featuresNO, lamda) 
    computeNumericalGradient(tmpCofi, x, theta)
    J, grad = cofiCostFunc(x, theta, y, r, userNO, moviesNO, featuresNO, lamda)
    return 1
'''
def loadMovieList(): 
    movieList = {}
    i = 0
    data = io.open("movie_ids.txt", "r", encoding="ISO-8859-14")
    for movie in data:
        movieList[i] = movie
        i= i+1
    #print(movieList)
    return movieList

def normalizeRatings(y, r):
    '''
    No need to add deviation, the range is really small.
    '''
    m, n = y.shape
    yMean = np.zeros((m))
    yNorm = np.zeros(y.shape)
    for i in range(m):
        tmp =  np.where(r[i,:]==1)
        yMean[i] = np.mean(y[i, tmp], axis=1)
        yNorm[i, tmp] = y[i, tmp] - yMean[i]
    return yNorm, yMean
