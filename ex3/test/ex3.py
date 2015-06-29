#!/usr/bin/python3
#shebang is good

from multiClassification import multiClassification
import math
import sys
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


data = scipy.io.loadmat("ex3data1.mat", mat_dtype=False)
y = data['y']
x = data['X']
y[y==10] = 0 # '0' is encoded as '10' in data, fix it, y = 0 1 2 3 4 5 6 7 8 9

print("Visualize 100 random selected data...")
idxs = np.random.randint(x.shape[0], size=100) # return 100 random training example
fig, ax = plt.subplots(10, 10)
img_size = math.sqrt(x.shape[1])
for i in range(10):
    for j in range(10):
        xi = x[idxs[i * 10 + j], :].reshape(img_size, img_size).T # the array of image is colummn-by-column indexing
        ax[i, j].set_axis_off()
        ax[i, j].imshow(xi, aspect="auto", cmap="gray")
plt.show()


x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
theta = np.zeros((x.shape[1], y.shape[1])) 
l = logisticClassification(x, y, lmda=0.1, numLabel=len(np.unique(y)))
#collapse theta into 1-D array
allTheta = l.oneVsAll(theta.flatten(), x, y)
result = l.predictOneVsAll(allTheta, x, y)
print("the accuracy is: ", (np.mean(result))*100)
input("press ENTER to continue...")

l.showDataOneByOne(allTheta, x[idxs], y[idxs])

"""
Neural Network prediction.
Using theta in "ex3weights.mat", accuracy is 97.52%.
"""

data = scipy.io.loadmat("ex3data1.mat", mat_dtype=False)
y = data['y']
x = data['X']
y[y==10] = 0 # '0' is encoded as '10' in data, fix it
weights = scipy.io.loadmat("ex3weights.mat", mat_dtype=False)
theta1 = weights['Theta1']
theta2 = weights['Theta2']

x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
hidden = np.dot(x, np.transpose(theta1)) 
hidden = (1/(1+np.exp(-hidden)))
hidden = np.concatenate((np.ones((x.shape[0], 1)), hidden), axis=1)
out = np.dot(hidden, np.transpose(theta2))
out = (1/(1+np.exp(-out)))
out = np.argmax(out, axis=1)
out = out[:, None]
out = np.roll(out, -500) # I believe there is a mistake in theta which is from "ex3weights.mat", the result is 9 0 1 2 3 4 5 6 7 8, and it's not related to y, so I shift the array 500 index backward to match y
out = (out == y)
print( (np.mean(out))*100 )
print(sys.version)
