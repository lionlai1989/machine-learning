#!/usr/bin/python3
#shebang is good

import logisticClassification as lc
import sys
import os
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import matplotlib as mpl
from plotly.plotly import *
from plotly.graph_objs import *
from types import *
from mpl_toolkits.mplot3d import Axes3D
import random


#change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))

"""
machine learning ex2 assignment
Logistic Regression
"""

data = np.loadtxt("ex2data1.txt", delimiter = ',', unpack = False)
x = data[:, 0:2] # x is input matrix
y = data[:, 2] # y is output matrix
y = y[:, None] # cast y from 1D array (100, ) into 2D array (100,1)

pos = np.where(y==1)
neg = np.where(y==0)
plt.scatter(x[pos, 0], x[pos, 1], marker='o', c='b')
plt.scatter(x[neg, 0], x[neg, 1], marker='x', c='r')

theta = np.zeros((x.shape[1]+1, y.shape[1]))
l = lc.logisticClassification(x, y)

xopt = fmin(l.costFunction, theta, args=(x, y))
plt.plot(x[:,0],-(np.dot(np.ones((x.shape[0], 1)).flatten(),xopt[0])+np.dot(x[:,0],xopt[1]))/xopt[2])
print(xopt)

ex=np.array([1, 45, 85])
print("Exam 1 score of 45, and Exam 2 score of 85, the probability= ", l.sigmoid(np.dot(ex, xopt)))
plt.show()

print(sys.version)
