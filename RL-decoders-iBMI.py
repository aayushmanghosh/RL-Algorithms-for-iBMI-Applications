'''Author: Aayushman Ghosh
   Department of Electrical and Computer Engineering
   University of Illinois Urbana-Champaign
   (aghosh14@illinois.edu)
   
   Version: v1.1
''' 

# Importing the necessary libraries and modules.
from utils import *
from algorithms import *
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
import FileBrowser
import argparse

parser = argparse.ArgumentParser()
scaler = StandardScaler()

# Adding arguments
parser.add_argument('dir', type = str, help = "directory of dataset")
parser.add_argument('expt', type = str, help = "name of the particular experiment")
parser.add_argument('--spars', type = int, default = 0, type = "Sparsity Rate")
parser.add_argument('--error', type = int, default = 0, help = "Error Rate")
parser.add_argument('--gamma', type = int, default = 0.0001, help = "Gamma value")
parser.add_argument('--muH', type = int, default = 0.01, help = "Hidden Layer Learning rate")
parser.add_argument('--muO', type = int, default = 0.01, help = "Hidden Layer Learning rate")
parser.add_argument('num_nodes', type = list, help = "Number of modes")

args = parser.parse_args()

# Defining the Global Variables --> Directory, error and sparsity error.
absolute_path = os.path.dirname(os.path.abspath('__file__'))
relative_path = args.dir 
expt = args.expt # subject to change depending on the experiment you are trying to execute
directory = os.path.join(absolute_path,os.path.join(relative_path, expt))
files = FileBrowser.uigetfile() 
error = args.error # Defining the error in Feedback
sparsity_rate = args.spars # Sparsity in the Feedback signals
gamma = args.gamma
muH = args.muH
muO = args.muO
num_nodes = args.num_modes
alpha = 0.1
beta = 0.1
gamma_AGREL = 0.02
num_nodes_AGREL = [1000,4]
epsilon = 0.01 # Exploration rate
gamma_DQN = 0.1 # Discount Factor

acc = []
error_acc = []
interval = []

def match_case(case):
    cases = {
        'a': banditron,
        'b': banditronRP,
        'c': AGREL,
        'd': HRL,
        'e': DQN,
        'f': QLGBM
    }
    # Get the function associated with the case and call it
    if case in cases:
        return cases[case]()
    else:
        return "Case not found"

def analysis(choice, dir, files, flag, **kwargs):
  for file in files:
    data = loadmat(os.path.join(dir,file))
    feature_mat = data["feature_mat"]
    X = feature_mat[:,:-1]
    y = feature_mat[:,-1]//90
    if choice == 'a':
       for error, sparsity_rate, gamma in kwargs.items():
          model = match_case(choice)
          pred = model(X, y, error, sparsity_rate, gamma=gamma)
    elif choice == 'b':
       for error, sparsity_rate, gamma in kwargs.items():
          model = match_case(choice)
          pred = model(X, y, 128, error, sparsity_rate, gamma=gamma)
    elif choice == 'c':
       for error, sparsity_rate, muH, muO, num_modes in kwargs.items():
          model = match_case(choice)
          pred = model(X, y, muH, muO, num_nodes, error, sparsity_rate)
    elif choice == 'd':
       for gamma, alpha, beta, num_nodes, error, sparsity_rate in kwargs.items():
           pred = model(X, y, gamma, alpha, beta, num_nodes, error, sparsity_rate)
    elif choice == 'e':
        target = to_categorical(y,4)
        for epsilon,gamma,error,sparsity_rate in kwargs.items():
            pred = model(X,target,epsilon,gamma,error,sparsity_rate)
    elif choice == 'f':
        target = to_categorical(y,4)
        for epsilon,gamma,error,sparsity_rate in kwargs.items():
            pred = model(X,target,epsilon,gamma,error,sparsity_rate)

    results = np.vstack([y*90,np.array(pred)*90]).T
    results_df = pd.DataFrame(results,columns=["True","Pred"])
    error_intervals = 1/(np.sum(results_df.loc[:,"True"] == results_df.loc[:,"Pred"])/results_df.shape[0]*100)
    error_acc.append(error_intervals)
    interval.append(2.58*np.sqrt(error_intervals*(1-error_intervals))/results_df.shape[0])
    acc.append((np.sum(results_df.loc[:,"True"] == results_df.loc[:,"Pred"])/results_df.shape[0])*100)

    if flag == 1:
        plt.figure(figsize=(10,5))
        plt.plot(range(1,len(files)+1),acc,'b-o')
        plt.grid()
        plt.ylim((20,120))
        plt.ylabel('Accuracy (in %)')
        plt.xlabel('Sessions')
        plt.xticks(range(1,len(files)+1), labels=['session_'+str(i) for i in range(1,len(files)+1)])
        for x,y in zip(range(1,len(files)+1),acc):
            label = "{:.2f}".format(y)
            plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,-20)) # distance from text to points (x,y)
        
        plt.title(directory.split('/')[-1]+str(model))
        plt.show()
        
    return acc        

# Getting the decoding accuracy for each algorithm
acc_Banditron = analysis('a', dir, files, 1, error, sparsity_rate, gamma=gamma)
acc_BanditronRP = analysis('b', dir, files, 1, error, sparsity_rate, gamma=gamma)
acc_HRL = analysis('c', dir, files, 1, muH, muO, num_nodes, error, sparsity_rate)
acc_AGREL = analysis('d', dir, files, 1, gamma_AGREL, alpha, beta, num_nodes_AGREL, error, sparsity_rate)
acc_DQN = analysis('e', dir, files, 1, epsilon, gamma_DQN, error, sparsity_rate)
acc_QLGBM = analysis('f', dir, files, 1, epsilon, gamma_DQN, error, sparsity_rate)

# Plotting the Decoding accuracy of all the Algorithms
plt.figure(figsize=(10,5))

plt.plot(range(1,len(files)+1),acc_Banditron,'b--o')
plt.plot(range(1,len(files)+1),acc_BanditronRP,'r--o')
plt.plot(range(1,len(files)+1),acc_HRL,'k--o')
plt.plot(range(1,len(files)+1),acc_AGREL,'g--o')
plt.plot(range(1,len(files)+1),acc_DQN,'y--o')
plt.plot(range(1,len(files)+1),acc_QLGBM,'--o')

plt.legend(['Banditron','BanditronRP','HRL','AGREL','Deep Q-Learning','LightGBM based Q-Learning'])

plt.grid()
plt.ylim((20,120))
plt.ylabel('Accuracy (in %)')
plt.xlabel('Sessions')
plt.xticks(range(1,len(files)+1), labels=['session_'+str(i) for i in range(1,len(files)+1)])
                 
plt.title(directory.split('/')[-1]+' Performance plots ')
#plt.savefig(directory.split('/')[-1]+' Performance plots.jpg')
plt.show()

# For taking out the accuracy data
acc_df = pd.DataFrame([acc_Banditron,acc_BanditronRP,acc_HRL,acc_AGREL,acc_DQN,acc_QLGBM]).T
acc_df.rename(dict(enumerate(['Banditron','BanditronRP','HRL','AGREL','Deep Q-Learning','QLGBM'])),axis=1,inplace=True)
acc_df.to_csv(directory.split('/')[-1]+' any-name-you-prefer.csv') # subject to change
