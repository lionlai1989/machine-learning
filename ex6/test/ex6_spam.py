#!/usr/bin/env python3

import sys
import os
import supportVectorMachine as svm
import scipy.io
from scipy.optimize import fmin_cg
import numpy as np
import matplotlib.pyplot as plt

# change python current working directory to the current folder
os.chdir(os.path.dirname(sys.argv[0]))
#print(os.getcwd())

########## Part 1: Email Preprocessing ########## 
print('Preprocessing sample email (emailSample1.txt)')
file_contents = ''
with open('emailSample1.txt', 'r') as file_contents:
    email = file_contents.read()
word_indices = svm.processEmail(email)
print(word_indices)
input('Program paused. Press enter to continue...')

########## Part 2: Feature Extraction ########## 

########## Part 3: Train Linear SVM for Spam Classification ########## 
 
########## Part 4: Test Spam Classification ########## 

########## Part 5: Top Predictors of Spam ########## 

########## Part 6: Try Your Own Emails########## 

print(sys.version)

