# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:51:53 2024

@author: 44754
"""

#import packages and functions
import numpy as np
import matplotlib.pyplot as plt
#import files
from utilis import *
from Initialize_weight import *
from ThreeLayerNN import *

###############################################################################           

#plot settings
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=18)  # fontsize of the figure title

###############################################################################   
"Hyperparameters for neural network"
#number of iterations
niter = 15000
#learning rate of the gradient descent
LR = 0.01
#activation fun of hidden layer
funhid = ReLu
#activation fun of output layer
funout = sigmoid
#gradient fun
gradfun = gradReLu
#the method we generate the initial condition of parameters
ini_name = 'he'

###############################################################################   
#generate the training and test sets
xtrain,ytrain,xtest,ytest = load_dataset()
#set dimensions of each layers
layer_dim = [xtrain.shape[0],10,5,1]
#update the parameters using DL_NN
para,J = DeepL_NN(xtrain,ytrain,layer_dim,ReLu,sigmoid,gradReLu,niter,LR,ini_name)
#predition on training set
ptrain = prediction(para,xtrain,layer_dim,funhid,funout)
accuracy(ptrain,ytrain,'training')

#predition on training set
ptest = prediction(para,xtest,layer_dim,funhid,funout)
accuracy(ptest,ytest,'test')
"""
#visualize training data set
plt.figure()
plt.scatter(xtrain[0,:],xtrain[1,:],c=ytrain,s=40,cmap=plt.cm.Spectral)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
"""

