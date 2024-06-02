# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:03:55 2024

@author: 44754
"""

from utilis import *
from NN_L2 import *
from NN_dropout import *

############################################################################
#generate the training set
xtrain,ytrain,xtest,ytest = load_2D_dataset()
"Hyperparameters settings"
#units in each layer (including the input layer)
layer_dim = [xtrain.shape[0],20,5,1]
#regularization parameter
Lambda = 0.01
#number of iterations in gradient descent
niter = 2000
#learning rate in gradient descent
LR = 0.1
#probability of keeping the nuron in the layer
keep_prob = 0.70
#activation functions in hidden layers
funhid = ReLu
#activation functions in hidden layers
funout = sigmoid
#gradient function for hidden layers
gradfun = gradReLu

############################################################################

"experiment with L2 regularization"
#update the parameters
para,J = DeepL_NN(xtrain,ytrain,layer_dim,funhid,funout,gradfun,niter,LR,Lambda)  
#prediction using updated parameters on training set
print('------Experiments with L2 Reg-------')
ptrain = prediction(para,xtrain,layer_dim,funhid,funout)
accuracy(ptrain,ytrain,'training')
#prediction using updated parameters on test set
ptest = prediction(para,xtest,layer_dim,funhid,funout)
accuracy(ptest,ytest,'test')

############################################################################

"experiments with dropout regularization"
#update the parameters and compute the cost of each iteration
para,J = NN_drop(xtrain,ytrain,layer_dim,funhid,funout,gradfun,niter,LR,keep_prob)
#prediction using updated parameters on training set
print('------Experiments with Dropout Reg-------')
ptrain = prediction(para,xtrain,layer_dim,funhid,funout)
accuracy(ptrain,ytrain,'training')
#prediction using updated parameters on test set
ptest = prediction(para,xtest,layer_dim,funhid,funout)
accuracy(ptest,ytest,'test')
