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

def estimateGaussian(x):
    mu = np.mean(x, axis=0) # mu is an array of n
    sigma2 = np.var(x, axis=0) # sigma2 is an array of n
    return mu, sigma2

def multivariateGaussian(x, mu, sigma2): 
    m, n = x.shape
    tempX = x-mu
    sigma = np.diag(sigma2)
    #sigma = np.dot(tempX.T, tempX) / m In theory, I think this line is 
    #the correct equation, but the result is not identical with ex8.py
    #when applying this line.
    p = ((2*np.pi)**(-n/2)) * (np.linalg.det(sigma)**-0.5) \
            * np.exp(-0.5*np.sum(np.multiply \
            ((np.dot(tempX, np.linalg.pinv(sigma))), tempX),axis=1))
    #p = ((2*np.pi)**(-n/2)) * (np.linalg.det(sigma)**-0.5) \
    #        * np.exp(-0.5*np.sum(np.dot \
    #        ((np.dot(tempX, np.linalg.pinv(sigma))), tempX.T),axis=1)) In 
    #theory, I think this line is the correct equation, but the result is 
    #not identical with ex8.py when applying this line.

    return p

def originalGaussian(x, mu, sigma2):
    p = (1/(np.sqrt(2*np.pi*sigma2))) * np.exp(-0.5*(x-mu)*(x-mu)/sigma2)
    p = np.prod(p, axis=1)
    return p

def visualizeFit(x, mu, sigma2):
    X1 = np.mgrid[0:35.5:0.5]
    X2 = np.mgrid[0:35.5:0.5]
    X1, X2 = np.meshgrid( X1, X2 )
    Z = multivariateGaussian( np.c_[X1.T.ravel().T, X2.T.ravel().T], mu, sigma2 )
    Z = Z.reshape( X1.shape[0], X1.shape[1] ).T
    a = np.arange( -20, 0, 3 )
    b = np.ones( a.shape ) * 10
    plt.contour(X1, X2, Z, pow( b, a ) )
    return 1

def selectThreshold(yVal, pVal):
    #print(yVal.shape, pVal.shape)
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    stepSize = (max(pVal) - min(pVal)) / 1000
    #print(max(pVal), min(pVal), stepSize)
    for epsilon in np.arange(min(pVal), max(pVal), stepSize):
        predict = np.less_equal(pVal, epsilon)
        predict = predict[:, None] # 1-D to 2-D, make it equal to  yVal
        a = np.where(yVal==True)
        b = np.where(yVal==False)
        tp = np.count_nonzero(np.equal(yVal[a[0]], predict[a[0]]))
        tn = np.count_nonzero(np.equal(yVal[b[0]], predict[b[0]]))
        fn = a[0].shape[0] - tp 
        fp = b[0].shape[0] - tn
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        F1 = 2*prec*rec/(prec+rec)
        #print(tp, tn, fn, fp)
        
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1
