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

"""machine learning ex4 assignment"""

def displayData(x, num):
    '''
    It randomly pick num data in x and displayed 2D data in a nice grid.
    '''
    print("Visualize", num, "random selected data...")
    idxs = np.random.randint(x.shape[0], size=num) # return num random training example
    tmp = np.sqrt(num)
    num = tmp.astype(np.int64)
    fig, ax = plt.subplots(num, num)
    img_size = math.sqrt(x.shape[1])
    for i in range(num):
        for j in range(num):
            xi = x[idxs[i * num + j], :].reshape(img_size, img_size).T # the array of image is colummn-by-column indexing
            ax[i, j].set_axis_off()
            ax[i, j].imshow(xi, aspect="auto", cmap="gray")
    plt.show()
    return ax

def sigmoid(z):
    '''
    It return a sigmoid of an input array.
    '''
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    '''
    It return a sigmoid gradient of an input array.
    '''
    return sigmoid(z)*(1-sigmoid(z))

def feedForward(x, theta1, theta2):
    '''
    Given input x, theta1 and theta2, it computes feedForward of neural network.
    '''
    a1 = x
    z2 = np.dot(a1, np.transpose(theta1))
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((x.shape[0], 1)), a2), axis=1)
    z3 = np.dot(a2, np.transpose(theta2))
    a3 = sigmoid(z3)
    return a1, z2, a2, z3, a3

def recodeLabel(y):
    '''
    It recodes array y to a proper shape which complied with H(x)
    '''
    labelNO = len(np.unique(y))
    out = np.zeros((y.shape[0], labelNO)) # (5000, 10)
    for i in range(y.shape[0]):
        out[i, y[i]] = 1
    out = np.roll(out, -1, axis=1) # shift 1 cloumn to left, makes 0 at the last column, the note should define the outout of node corresponded to 0~9
    return out

def computeCost(theta, inputLayerSize, hiddenLayerSize, numLabel, 
        x, y, lamda):
    '''
    For some reason(matlab is different from python), we have to define computeCost separately, which can be used in fmin_cg.
    '''
    theta1 = np.reshape(theta[:(inputLayerSize+1)*hiddenLayerSize], (hiddenLayerSize, inputLayerSize+1))
    theta2 = np.reshape(theta[(inputLayerSize+1)*hiddenLayerSize:], (numLabel, hiddenLayerSize+1))
    m, n = x.shape
    a1, z2, a2, z3, a3 = feedForward(x, theta1, theta2)
    yk = recodeLabel(y)
    
    leftTerm = np.sum( np.multiply(np.log(a3), -yk) - np.multiply(np.log(1-a3),1-yk) )/m
    rigntTerm = (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))*lamda/(2*m)
    cost = leftTerm + rigntTerm
    return cost

def computeGradient(theta, inputLayerSize, hiddenLayerSize, numLabel, 
        x, y, lamda):
    '''
    For some reason(matlab is different from python), we have to define computeGradient separately, which can be used in fmin_cg.
    '''
    theta1 = np.reshape(theta[:(inputLayerSize+1)*hiddenLayerSize], (hiddenLayerSize, inputLayerSize+1))
    theta2 = np.reshape(theta[(inputLayerSize+1)*hiddenLayerSize:], (numLabel, hiddenLayerSize+1))
    m, n = x.shape
    a1, z2, a2, z3, a3 = feedForward(x, theta1, theta2)
    yk = recodeLabel(y)

    d3 = a3 - yk
    tmp = np.dot(d3, theta2)
    d2 = np.multiply(tmp[:,1:], sigmoidGradient(z2) )

    accum1 = np.dot(d2.T, a1)
    accum2 = np.dot(d3.T, a2)
    tmp1=theta1*lamda/m
    tmp1[:,0]=0
    tmp2=theta2*lamda/m
    tmp2[:,0]=0
    theta1Grad = accum1/m + tmp1
    theta2Grad = accum2/m + tmp2
    thetaGrad = np.concatenate((theta1Grad.flatten(), theta2Grad.flatten()))
    return thetaGrad

def nnCostFunction(theta, inputLayerSize, hiddenLayerSize, numLabel, 
        x, y, lamda):
    '''
    It returns cost function and gradient of cost function.
    '''
    theta1 = np.reshape(theta[:(inputLayerSize+1)*hiddenLayerSize], (hiddenLayerSize, inputLayerSize+1))
    theta2 = np.reshape(theta[(inputLayerSize+1)*hiddenLayerSize:], (numLabel, hiddenLayerSize+1))
    m, n = x.shape
    a1, z2, a2, z3, a3 = feedForward(x, theta1, theta2)
    yk = recodeLabel(y)
    
    leftTerm = np.sum( np.multiply(np.log(a3), -yk) - np.multiply(np.log(1-a3),1-yk) )/m
    rigntTerm = (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))*lamda/(2*m)
    cost = leftTerm + rigntTerm

    d3 = a3 - yk
    tmp = np.dot(d3, theta2)
    tmp = tmp[:,1:]
    d2 = np.multiply(tmp , sigmoidGradient(z2) )

    accum1 = np.dot(d2.T, a1)
    accum2 = np.dot(d3.T, a2)
    tmp1=theta1*lamda/m
    tmp1[:,0]=0
    tmp2=theta2*lamda/m
    tmp2[:,0]=0
    theta1Grad = accum1/m + tmp1
    theta2Grad = accum2/m + tmp2
    thetaGrad = np.concatenate((theta1Grad.flatten(), theta2Grad.flatten()))
    return cost, thetaGrad

def randInitializeWeight(theta1, theta2, epislon):
    '''
    It randomly initializes theta1 and theta2 in the gap of epislon.
    '''
    a,b = theta1.shape
    c,d = theta2.shape
    theta1 = theta1.flatten('C')
    theta2 = theta2.flatten('C')
    theta1 = np.random.rand(theta1.shape[0])*2*epislon-epislon
    theta2 = np.random.rand(theta2.shape[0])*2*epislon-epislon
    theta1 = theta1.reshape(a, b)
    theta2 = theta2.reshape(c, d)
    return theta1, theta2

def debugInitializeWeights(fanOut, fanIn):
    '''
    It initializes the weights of a layer with fanIn incoming connections 
    and fanOut outgoing connections.
    '''
    w = np.zeros((fanOut, 1+fanIn))
    w = np.reshape(np.sin(np.arange(w.size)), w.shape)/10
    return w

def computeNumericalGradient(theta, inputLayerSize, hiddenLayerSize, numLabel, 
        x, y, lamda):
    '''
    It computes the gradient using "finite differences" and gives us a numerical estimate of the gradient.
    '''
    numGrad = np.zeros((theta.shape))
    perTurb = np.zeros((theta.shape))
    e = 1e-4
    for i in range(len(theta)):
        perTurb[i] = e
        loss1 = nnCostFunction(theta - perTurb, inputLayerSize, hiddenLayerSize, numLabel, x, y, lamda)[0]
        loss2 = nnCostFunction(theta + perTurb, inputLayerSize, hiddenLayerSize, numLabel, x, y, lamda)[0]
        numGrad[i] = (loss2-loss1)/(2*e)
        perTurb[i] = 0
    return numGrad

def checkNNGradients(lamda):
    '''
    It creates a small neural network to check the backpropagation gradients.
    '''
    inputLayerSize = 3
    hiddenLayerSize = 5
    numLabel = 3
    m = 5
    theta1 = debugInitializeWeights(hiddenLayerSize, inputLayerSize)
    theta2 = debugInitializeWeights(numLabel, hiddenLayerSize)
    x = debugInitializeWeights(m, inputLayerSize-1)
    y = np.transpose(np.mod(np.arange(m), numLabel))
    y = y[:,None]
    theta = np.concatenate((theta1.flatten(), theta2.flatten()))
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    cost, grad = nnCostFunction(theta, inputLayerSize, hiddenLayerSize, numLabel, x, y, lamda)
    numGrad = computeNumericalGradient(theta, inputLayerSize, hiddenLayerSize, numLabel, x, y, lamda)
    diff = np.linalg.norm(numGrad-grad)/np.linalg.norm(numGrad+grad)
    return diff

def predict(theta1, theta2, x, y):
    '''
    It calculates the result of feedForward, and returns the result compared to y.
    '''
    a1, z2, a2, z3, a3 = feedForward(x, theta1, theta2)
    tmp = np.argmax(a3, axis=1)
    tmp = tmp[:, None]
    tmp = np.roll(tmp, -500)
    #compare the predicted and the training example
    result = (tmp == y) 
    return result
