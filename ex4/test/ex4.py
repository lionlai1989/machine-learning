#!/usr/bin/python3
#shebang is good

import sys
import os
import neuralNetworksLearning as nn
import math
import numpy as np
import scipy.io 
from types import *
from mpl_toolkits.mplot3d import Axes3D
import random

from scipy.optimize import fmin
from scipy.optimize import fmin_cg
from scipy.optimize import fmin_bfgs

import matplotlib.pyplot as plt
import matplotlib as mpl

from plotly.plotly import *
from plotly.graph_objs import *

#change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))
#print(os.getcwd())

data = scipy.io.loadmat("ex4data1.mat", mat_dtype=False)
x = data['X']
y = data['y']
y[y==10] = 0 # '0' is encoded as '10' in data, fix it, y = 0 1 2 3 4 5 6 7 8 9
print(x.shape, y.shape)

data1 = scipy.io.loadmat("ex4weights.mat", mat_dtype=False)
theta1 = data1['Theta1']
theta2 = data1['Theta2']
print(theta1.shape, theta2.shape)

x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
nn.nnCostFunction(x, y, theta1, theta2, lamda=1)
theta1, theta2 = nn.randInitializeWeight(theta1, theta2, epislon=0.12)

print(sys.version)
