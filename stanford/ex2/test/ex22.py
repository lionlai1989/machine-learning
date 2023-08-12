#!/usr/bin/python3
#shebang is good

import logisticClassification as lc
import sys
import os
import numpy as np
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs

import matplotlib.pyplot as plt
import matplotlib as mpl
from plotly.plotly import *
from plotly.graph_objs import *
from types import *
from mpl_toolkits.mplot3d import Axes3D
import random


#change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))

"""machine learning ex2 assignment"""

        

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""the assignment start from here"""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
data = np.loadtxt("ex2data2.txt", delimiter = ',', unpack = False)
x = data[:, 0:2] # x is input matrix
y = data[:, 2] # y is output matrix
y = y[:, None] # cast y from 1D array (100, ) into 2D array (100,1)

pos = np.where(y==1)
neg = np.where(y==0)
plt.scatter(x[pos, 0], x[pos, 1], marker='o', c='b')
plt.scatter(x[neg, 0], x[neg, 1], marker='x', c='r')


#no need to do this, mapfeature do this for me
#x = np.concatenate((np.ones((trainingNO, 1)), x), axis=1)

l = lc.logisticClassification(x, y)
x = l.mapFeature(x[:,0],x[:,1])
theta = np.zeros((x.shape[1], y.shape[1])) 
#print(x.shape)

lmda = 1
print("initial cost function: ", 
        l.mapFeatureCostFn(theta, x, y, lmda))
xopt = fmin_bfgs(l.mapFeatureCostFn, theta, args=(x, y, lmda))
theta = xopt
print("cost function after optimiztion: ", 
        l.mapFeatureCostFn(theta, x, y, lmda))
l.plotDecisionBoundry(theta)
input("press ENTER to show the classification result")
plt.show()
print(sys.version)
