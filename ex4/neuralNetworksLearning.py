#!/usr/bin/env python3

import math
import sys
import numpy as np
import scipy.io 
from types import *
from mpl_toolkits.mplot3d import Axes3D
import random

from scipy.optimize import fmin_cg

import matplotlib.pyplot as plt
import matplotlib as mpl

from plotly.plotly import *
from plotly.graph_objs import *

"""machine learning ex4 assignment"""

def displayData(x, exampleWidth):
    return 1

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))
    #return np.multiply(sigmoid(z),(1-sigmoid(z)))

def feedForward(x, theta1, theta2):
    a1 = x
    z2 = np.dot(a1, np.transpose(theta1))
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((x.shape[0], 1)), a2), axis=1)
    z3 = np.dot(a2, np.transpose(theta2))
    a3 = sigmoid(z3)
    return a1, z2, a2, z3, a3

def recodeLabel(y):
    labelNO = len(np.unique(y))
    out = np.zeros((y.shape[0], labelNO)) # (5000, 10)
    for i in range(y.shape[0]):
        out[i, y[i]] = 1
    out = np.roll(out, -1, axis=1) # shift 1 cloumn to left, makes 0 at the last column, the note should define the outout of node corresponded to 0~9
    return out
    
def nnCostFunction(x, y, theta1, theta2, lamda): 
    m, n = x.shape
    a1, z2, a2, z3, a3 = feedForward(x, theta1, theta2)
    yk = recodeLabel(y)
    
    leftTerm = np.sum( np.multiply(np.log(a3), -yk) - np.multiply(np.log(1-a3),1-yk) )/m
    rigntTerm = (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))*lamda/(2*m)
    cost = leftTerm + rigntTerm
    print(cost)

    d3 = a3 - yk
    print(d3.shape)
    tmp = np.dot(d3, theta2)
    tmp = tmp[:,1:]
    print(tmp.shape)
    d2 = np.multiply(tmp , sigmoidGradient(z2) )
    print(d2.shape)

    accum1 = np.dot(d2.T, a1)
    accum2 = np.dot(d3.T, a2)
    theta1Grad = accum1/m
    theta2Grad = accum2/m

    print(accum1, accum2)

def randInitializeWeight(theta1, theta2, epislon):
    a,b = theta1.shape
    c,d = theta2.shape
    theta1 = theta1.flatten('C')
    theta2 = theta2.flatten('C')
    theta1 = np.random.rand(theta1.shape[0])*2*epislon-epislon
    theta2 = np.random.rand(theta2.shape[0])*2*epislon-epislon
    theta1 = theta1.reshape(a, b)
    theta2 = theta2.reshape(c, d)
    return theta1, theta2



