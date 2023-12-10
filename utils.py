'''Author: Aayushman Ghosh
   Department of Electrical and Computer Engineering
   University of Illinois Urbana-Champaign
   (aghosh14@illinois.edu)
   
   Version: v1.0
''' 

import os
import numpy as np
import scipy.special as sp
from scipy.io import loadmat

# Defining the sigmoid function.
def sigmoid(x):
  z = 1/(1+sp.expit(-x))
  return z

# Defining the ReLU function.
def ReLU(x):
  z = np.maximum(0,x)
  return z

# Defining the softmax function.
def softmax(x):
  x = x - np.max(x)
  z = np.exp(x)/np.sum(np.exp(x))
  return z



        