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

def svmTrain(x, y, C, kernelFunction, tol, max_passes, sigma):
    if sigma == 0:
        gamma = 0
    else:
        gamma = 1.0 / sigma
    clf = svm.SVC(C=C, tol=tol, max_iter=max_passes, kernel=kernelFunction, gamma=gamma)
    #clf = svm.SVC(C=C, kernel='gaussianKernel')
    return clf.fit(x, y.ravel())


def visualizeBoundaryLinear(x, y, model):
    kernel = model.get_params()['kernel']
    if kernel == 'linear':
        w = model.dual_coef_.dot(model.support_vectors_).flatten()
        xp = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
        print(w, xp.shape)
        yp = (-w[0] * xp - model.intercept_) / w[1]
        print(model.intercept_)
        #plt.plot(xp, yp, 'b-')
    elif kernel == 'rbf':
        x1plot = np.linspace( min(x[:, 0]), max(x[:, 0]), 100)
        x2plot = np.linspace( min(x[:, 1]), max(x[:, 1]), 100)
        x1, x2 = np.meshgrid(x1plot, x2plot)
        vals = np.zeros(np.shape(x1))
        for i in range(0, np.shape(x1)[1]):
            this_x = np.c_[ x1[:, i], x2[:, i] ]
            vals[:, i] = model.predict(this_x)

    return 1

def dataset3Params(x, y, xVal, yVal):

    return 1, 1


def linearKernel(x1, x2):
    '''
    linearKernel returns a linear kernel between x1 and x2
    '''
    # Ensure that x1 and x2 are column vector
    return np.dot(np.transpose(x1), x2)

def gaussianKernel(x1, x2, sigma):
    '''
    gaussianKernel returns a radial function kernel between x1 and x2
    '''
    return np.exp(-sum((x1-x2)**2)/(2*(sigma**2)))
