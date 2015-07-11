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
    tmp = np.dot(x, theta)-y
    theta[0,0] = 0
    cost = np.sum(np.multiply(tmp, tmp))/(2*m) + np.sum(np.multiply(theta, theta))*lamda/(2*m)
    grad = ((np.dot(np.transpose(x), tmp))/(m))+(theta*(lamda)/(m))
    # grad.flatten makes grad to be a 1-D array, and it can help when used in fmin_cg
    return cost, grad.flatten()

def trainLinearReg(x, y, lamda):
    m, n = x.shape
    theta = np.zeros((n, 1))
    costFunction = lambda  theta: linearRegCostFunction(x, y, np.copy(theta), lamda)[0] # AGAIN, np.copy(theta), np.copy here is fucking important, pass by value not by object.
    costGrad = lambda theta: linearRegCostFunction(x, y, np.copy(theta), lamda)[1]
    #when using fmin_cg, the initial value must be 1-D array
    xopt = fmin_cg(costFunction, np.copy(theta), fprime=costGrad, gtol=1e-09, maxiter=200)
    return xopt

def learningCurve(x, y, xv, yv, lamda):
    #print(x, y, xv, yv)
    m, n = x.shape
    errTrain = np.zeros((m, 1))
    errVal = np.zeros((m, 1))
    print(x.shape, xv.shape)
    for i in range(m):
        theta = trainLinearReg(x[0:i+1,:],y[0:i+1,:], lamda)
        theta = theta[:,None]
        errTrain[i] = linearRegCostFunction(x[0:i+1,:], y[0:i+1,:], 
                                            np.copy(theta), lamda)[0]
        errVal[i] = linearRegCostFunction(xv, yv, np.copy(theta), lamda)[0]
        print(theta)
    print(errTrain, errVal) 
    return errTrain, errVal

def polyFeatures(x, p):

    return 1




