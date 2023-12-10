'''Author: Aayushman Ghosh
   Department of Electrical and Computer Engineering
   University of Illinois Urbana-Champaign
   (aghosh14@illinois.edu)
   
   Version: v1.0
''' 

from utils import *
import warnings
import time
warnings.filterwarnings('ignore')
import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.activations import relu
from keras import Model, Input
from keras.losses import binary_crossentropy
from keras.metrics import Accuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, Sequential
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import scipy.special as sp

# Defining the Banditron Function (Single Layered Network)
'''
Here, **kwargs is used to denote the arbitary input functions. The error and sparse_rate
corresponds to the error and the sparsity introduced to the feedback signal (refer to the 
paper to understand the physical significance). k denotes the number of classes and is 
fixed at 4 for our experiments. X denotes the spike count data (observation -- input dataset) 
and y denotes the true labels associated with each observation (X) 
'''
def banditron(X, y, error, sparse_rate, k=4, **kwargs):
    T = X.shape[0] 
    d = X.shape[1]
    np.random.seed(100) 
    W = np.zeros((k, d)) # Initializing the Weight Matrix. 
    error_count = np.zeros(T) 
    pred = [] # The predicted labels will be stored here.
    
    # The exploration exploitation constant and eta are given as optional arguments.
    if "gammas" not in kwargs:
        gammas = [kwargs["gamma"] for i in range(T)]
    else:
        gammas = kwargs["gammas"]
    
    if "eta" in kwargs:
        eta = kwargs["eta"]
    
    # Evaluative framework (refer to the paper to understand the mathematics)
    for t in range(T):
        gamma = gammas[t]
        y_hat = np.argmax(np.dot(W, X[t]))
        p = [gamma/k for i in range(k)]
        p[y_hat] = p[y_hat] + 1 - gamma
        y_tilde = np.random.choice(range(k), p=p)
        pred.append(y_tilde)
        sparsify = np.random.choice([True,False],p=[sparse_rate,1-sparse_rate])
        if not sparsify:
          if y_tilde != y[t]:
            choice = np.random.choice(range(2),p=[error,1-error])
            if choice == 1:
              W[y_hat] = W[y_hat] - X[t]   
            else:
              W[y_hat] = W[y_hat] - X[t]
              W[y_tilde] = W[y_tilde] + X[t] / p[y_tilde]         
          else:
            choice = np.random.choice(range(2),p=[error,1-error])
            if choice == 1:
              W[y_hat] = W[y_hat] - X[t]
              W[y_tilde] = W[y_tilde] + X[t] / p[y_tilde]
            else:
              W[y_hat] = W[y_hat] - X[t]       
    return pred

# Defining the Banditron-RP Function (Three Layered Network)
'''
Here, **kwargs is used to denote the arbitary input functions. The error and sparse_rate
corresponds to the error and the sparsity introduced to the feedback signal (refer to the 
paper to understand the physical significance). k denotes the number of classes and is 
fixed at 4 for our experiments. X denotes the spike count data (observation -- input dataset) 
and y denotes the true labels associated with each observation (X) 
'''
def banditronRP(X, y, k, error, sparse_rate, **kwargs):
    d = X.shape[1]
    Wrand = np.random.uniform(size=(k,d)) # The random Weight matrix generated from a normal distribution.
    f = sigmoid(np.dot(Wrand,X.T)) # The non-linear projection vector input to the hidden layer.
    pred = banditron(f.T, y, error, sparse_rate, **kwargs) # f(t) = Sigmoid(Wrand.x(t)) is given as an input to the Banditron.
    return pred

# Defining the HRL function (Three Layered Network)
# Initializing the weight matrices
'''
In HRL the weight matrices are not initialized as zero matrix instead random floats
are extracted from Gaussian Distribution (mean = 0, var = 1). Here inp_shp denotes the
number of rows, and out_shp the number of columns for the weight matrix.
'''
def initialize(inp_shp,out_shp):
  W = np.random.randn(out_shp,inp_shp)
  return W

'''
The error and sparse_rate corresponds to the error and the sparsity introduced to the 
feedback signal (refer to the paper to understand the physical significance). muH and muO
denotes the learning rates corresponding to the weight updation policy. num_nodes is a matrix referring to 
the number of hidden nodes and output nodes. X denotes the spike count data (observation -- input dataset) 
and y denotes the true labels associated with each observation (X) 
'''  
def HRL(X, y, muH, muO, num_nodes, error, sparse_rate):
    T = X.shape[0]
    W = [0]*(len(num_nodes)-1)
    pred = []
    num_nodes[0] = num_nodes[0]+1
    
    # Initializing the weight matrices.
    for i in range(1,len(num_nodes)):
      W[i-1] = initialize(num_nodes[i-1],num_nodes[i])
    
    # Computing the output for the hidden layer and output layer.
    for t in range(T):
      x = np.insert(X[t],0,1)
      out = [x.reshape(-1,1)]*(len(num_nodes))     
      for i in range(1,len(num_nodes)):
        out[i] = np.tanh(np.dot(W[i-1],out[i-1]))
      
      # Evaluative framework (refer to the paper to understand the mathematics)
      out[-1] = np.tanh(np.dot(W[-1],np.sign(out[-2])))
      yhat = np.argmax(out[-1])
      sparsify = np.random.choice([True,False],p=[sparse_rate,1-sparse_rate])
      if not sparsify:
        if yhat == y[t]:
          f = np.random.choice([-1,1],p=[error,1-error])
        else:
          f = np.random.choice([1,-1],p=[error,1-error])
        dW = [0]*(len(num_nodes)-1)
  
        for i in range(1,len(num_nodes)):
          dW[i-1] = muH*f*(np.dot((np.sign(out[i])-out[i]),out[i-1].T)) + muH*(1-f)*(np.dot((1-np.sign(out[i])-out[i]),out[i-1].T))
          W[i-1] = W[i-1] + dW[i-1]

        dW[-1] = muO*f*(np.dot((np.sign(out[-1])-out[-1]),out[-2].T)) + muO*(1-f)*(np.dot((1-np.sign(out[-1])-out[-1]),out[-2].T))
        W[-1] = W[-1] + dW[-1]

      pred.append(yhat)
    return pred

# Defining the AGREL function (Three Layered Network)
# Initializing the weight matrices
'''
In AGREL the weight matrices are not initialized as zero matrix instead random floats
are extracted from Gaussian Distribution (mean = 0, var = 1). Here inp_shp denotes the
number of rows, and out_shp the number of columns for the weight matrix.
'''
def initialize(inp_shp,out_shp):
  W = np.random.uniform(low=-1,high=1,size=(out_shp,inp_shp))
  return W

'''
The error and sparse_rate corresponds to the error and the sparsity introduced to the 
feedback signal (refer to the paper to understand the physical significance). alpha, and beta
denotes the learning rates corresponding to the weight updation policy. num_nodes is a matrix referring to 
the number of hidden nodes and output nodes. gamma denotes the exploration-exploitation trade-off. 
X denotes the spike count data (observation -- input dataset) and y denotes the true labels associated with each observation (X) 
''' 
def AGREL(X, y, gamma, alpha, beta, num_nodes, error, sparse_rate):
    T = X.shape[0]
    pred = []
    
    # Initializing the weight matrices.
    W_H = initialize(num_nodes[0]+1,num_nodes[1])
    W_O = initialize(num_nodes[1],num_nodes[2])

    for t in range(T):
      x = np.insert(X[t],0,1).reshape(-1,1) 
      y_H = sigmoid(np.dot(W_H,x))
      Z = np.dot(W_O,y_H)
      y_O = softmax(Z)
      yhat = np.argmax(y_O)
      
      # Computing the output for the hidden layer and output layer.
      outs = np.zeros(Z.shape)
      outs[yhat] = 1
      explore = np.random.uniform()<gamma 
      
      # Evaluative framework (refer to the paper to understand the mathematics)
      if explore:
        y_tilde = np.random.randint(low=0,high=num_nodes[-1])
      else:
        y_tilde = yhat
      sparsify = np.random.choice([True,False],p=[sparse_rate,1-sparse_rate])
      if not sparsify:
        if y_tilde == y[t]:
          delta = np.random.choice([-1 ,1 - outs[y_tilde]],p=[error,1-error])
        else:
          delta = np.random.choice([-1 ,1 - outs[y_tilde]],p=[1-error,error])
        '''
        In AGREL, δ therefore influences plasticity through
        an expansive function f (δ), which also helps to fasten
        the learning process. f (δ) is defined such as, it takes
        large values if δ is close to 1, that is, when actions are
        rewarded unexpectedly.
        '''
        if delta >= 0:
          f = delta/(1-delta+1e-4) 
        else:
          f = delta
        # Weight updation policy of AGREL
        dW_O = beta*f*y_H.T
        dW_H = alpha*f*np.dot((y_H*(1-y_H)*W_O[y_tilde,:].reshape(-1,1)),x.T)

        W_O[y_tilde,:] = W_O[y_tilde,:] + dW_O
        W_H = W_H + dW_H
        
      pred.append(yhat)
    return pred

# Defining the DQN function (Four Layered Network)
# Defining the DQN model
'''
Here, the Deep Q Learning model is computed as a four layer network, where the number of
nodes in the input layer changes with the experiment, and is given by inp_shp. This 
following function sequentially builds the model, where the output layer has 4 classes,
and the two hidden layers has 128 neurons.
'''
def get_DQN_model(inp_shp, lr=0.01):
  np.random.seed(101)
  model = Sequential()
  model.add(Dense(128, activation='relu', input_shape=(inp_shp,))) # hidden layer1 neurons = 128
  model.add(Dense(128, activation='relu')) # hidden layer2 neurons = 128
  model.add(Dense(4, activation='linear')) # Output layer neurons = 4
  model.compile(loss='mse',optimizer='adam') # A mse loss function is used with an adam optimizer.
  return model

'''
The error and sparse_rate corresponds to the error and the sparsity introduced to the 
feedback signal (refer to the paper to understand the physical significance). epsilon and gamma 
denotes the exploration constant and discount factor. X denotes the spike count data 
(observation -- input dataset) and y denotes the true labels associated with each observation (X). 
'''
def DQN(X,Y,epsilon,gamma,error,sparse_rate):

  inp_shp = X.shape[1] # Number of electrodes --> corresponding to the no. of neurons in the first layer.
  model = get_DQN_model(inp_shp)
  T = X.shape[0]
  X_norm = StandardScaler.fit_transform(X)
  pred = []
  
  # Evaluative framework (refer to the paper to understand the mathematics)
  for t in range(T):
    x = X_norm[t,:].reshape(1,X.shape[1]).astype(np.float32)
    y = Y[t,:].reshape(1,Y.shape[1])
    Q = model.predict(x)

    yhat = np.argmax(Q)
    explore = np.random.uniform() < epsilon
    if explore:
      ytilde = np.random.randint(low=0,high=4)
    else:
      ytilde = yhat
    sparsify = np.random.choice([True,False],p=[sparse_rate,1-sparse_rate])
    if not sparsify:
      if ytilde == np.argmax(y):
        r = np.random.choice([-1 ,1],p=[error,1-error])
      else:
        r = np.random.choice([1 ,-1],p=[error,1-error])

      Target = r + gamma*np.amax(Q)
      Target_vec = Q
      Target_vec[0,ytilde] = Target
      Target_vec = np.array(Target_vec).reshape(1,4)

      model.fit(x,Target_vec,batch_size=1, epochs=1,verbose=0)

    pred.append(ytilde)
  return np.array(pred)

# Defining the LGBM based Q-Learning Function
# Building the LGBM Model
'''
Here, instead of using a deep neural network to build the DQN framework, we have 
used the LightGBM framework with a Q-Learning policy. We call this model the QLGBM Network.
'''
def get_QLGBM_model():
  model = MultiOutputRegressor(LGBMRegressor(n_jobs=-1)) 
  return model

'''
The error and sparse_rate corresponds to the error and the sparsity introduced to the 
feedback signal (refer to the paper to understand the physical significance). epsilon and gamma 
denotes the exploration constant and discount factor. X denotes the spike count data 
(observation -- input dataset) and y denotes the true labels associated with each observation (X). 
'''
def QLGBM(X,Y,epsilon,gamma,error,sparse_rate):
  model = get_QLGBM_model()
  T = X.shape[0]
  X_norm = StandardScaler.fit_transform(X)
  pred = []
  isFit = False
  
  # Evaluative framework (refer to the paper to understand the mathematics)
  for t in range(T-1):
    x = X_norm[t:t+2,:].astype(np.float32)
    y = Y[t:t+2,:]
    if isFit:
      Q = model.predict(x)
    else:
      np.random.seed(101)
      Q = np.random.uniform(low=-1,high=1,size=(2,4))
    yhat = np.argmax(Q[0,:])
    explore = np.random.uniform() < epsilon
    if explore:
      ytilde = np.random.randint(low=0,high=4)
    else:
      ytilde = yhat
    sparsify = np.random.choice([True,False],p=[sparse_rate,1-sparse_rate])
    if not sparsify:
      if ytilde == np.argmax(y):
        r = np.random.choice([-1 ,1],p=[error,1-error])
      else:
        r = np.random.choice([1 ,-1],p=[error,1-error])
      Target = r + gamma*np.amax(Q)
      Target_vec = Q
      Target_vec[0,ytilde] = Target
      Target_vec = np.array(Target_vec).reshape(2,4)
      model.fit(x,Target_vec)
      isFit = True
    pred.append(ytilde)
  return np.array(pred)