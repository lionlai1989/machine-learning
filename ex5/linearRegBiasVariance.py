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

def  linearRegCostFunction(x, y, theta, lamda):
    m, n = x.shape
    if(theta.ndim == 1):
        theta = theta[:, None]
    tmp = np.dot(x, np.copy(theta))-y
    theta[0,0] = 0
    cost = np.sum(np.multiply(tmp, tmp))/(2*m) + \
            (np.sum(np.multiply(np.copy(theta), np.copy(theta)))*lamda/(2*m))
    grad = ((np.dot(np.transpose(x), tmp))/(m))+(theta*(lamda)/(m))
    # grad.flatten makes grad to be a 1-D array, 
    # and it can help when used in fmin_cg
    return cost, grad.flatten()

def trainLinearReg(x, y, lamda):
    m, n = x.shape
    theta = np.zeros((n, 1))
    # AGAIN, np.copy(theta), np.copy here is fucking important,
    # pass by value not by object. 
    costFunction = lambda  theta: linearRegCostFunction(x, y, \
                                                        (theta), \
                                                        lamda)[0] 
    costGrad = lambda theta: linearRegCostFunction(x, y, \
                                                    (theta), \
                                                    lamda)[1]
    #when using fmin_cg, the initial value must be 1-D array
    xopt = fmin_cg(costFunction, np.copy(theta), \
                    fprime=costGrad, gtol=1e-09, maxiter=200)
    
    return xopt


def learningCurve(x, y, xv, yv, lamda):
    m, n = x.shape
    errTrain = np.zeros((m, 1))
    errVal = np.zeros((m, 1))
    for i in range(m):
        theta = trainLinearReg(x[0:i+1,:],y[0:i+1,:], lamda)
        theta = theta[:,None]
        errTrain[i] = linearRegCostFunction(x[0:i+1,:], y[0:i+1,:], \
                                            np.copy(theta), lamda)[0]
        errVal[i] = linearRegCostFunction(xv, yv, np.copy(theta), lamda)[0]
    return errTrain, errVal

def polyFeatures(x, p):
    m, n = x.shape
    polyX = np.zeros(shape=(m, p)) # edit here
    for i in range(p):
        polyX[:, i] = x[:, 0]**(i+1)
    return polyX

def featureNormalize(x):
    mean = np.mean(x, axis=0)
    tmp = (x - mean)
    # ddof is Delta Degrees of Freedom,
    # the divisor used in calculations is N - ddof
    std = np.std(tmp, axis=0, ddof=0) 
    normX = tmp/std
    return normX, mean, std

def validationCurve(polyX, y, polyXv, yv):
    lamdaVec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    errTrain = np.zeros((len(lamdaVec)))
    errVal = np.zeros((len(lamdaVec)))
    for i in range(len(lamdaVec)):
        theta = trainLinearReg(polyX, y, lamdaVec[i])
        errTrain[i] = linearRegCostFunction(polyX, y, np.copy(theta), lamdaVec[i])[0]
        errVal[i] = linearRegCostFunction(polyXv, yv, np.copy(theta), lamdaVec[i])[0]
    return lamdaVec, errTrain, errVal

def plotFit(xmin, xmax, mean, std, theta, p):
    x = np.arange(xmin-15, xmax+25, 0.05)
    x = x[:,None]
    polyX = polyFeatures(np.copy(x), p)
    polyX = (polyX - mean)/std
    polyX = np.concatenate((np.ones((polyX.shape[0], 1)), polyX), axis=1)
    plt.plot(x, np.dot(polyX, theta))
