# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:14:42 2024

@author: 44754
"""
import time
import matplotlib.pyplot as plt
from utilis import *
from GD import *
from GD_momentum import *
from GD_Adam import *

###############################################################################
#generate the random training set
x,y = load_dataset()
"Hyperparameters"
#dimensions in each layer (including the input layer)
layer_dim = [x.shape[0],5,2,1]
#number of epoch in the gradient descent algorithm
n_epoch = 10000
#learning rate in gradient descent
LR = 0.0007
#size of each mini-Batch
batch_size = 64
#activation function used in hidden layer
funhid = ReLu
#activation function used in output layer
funout = sigmoid
#gradient of the activation function in hidden layer
gradfun = gradReLu
#parameters for momentum and adam
beta1 = 0.9
beta2 = 0.999
#small fraction to avoid divide by zero
epsilon = 1e-8

###############################################################################

"experiment using mini-Batch gradient descent"
start_time = time.time()
para_GD,J_GD = Mini_BatchGD(x,y,layer_dim,n_epoch,batch_size,funhid,\
                            funout,gradfun,LR)
print("---Running GD uses %s seconds ---" % (time.time() - start_time))

###############################################################################
"experiment using mini-Batch gradient descent with momentum"
start_time = time.time()
para_mo,J_mo = Mini_Batch_GDmomentum(x,y,layer_dim,n_epoch,batch_size,funhid,funout,gradfun,LR,beta1)
print("---Running GD with momentum uses %s seconds ---" % (time.time() - start_time))

###############################################################################
"experiment using mini-Batch gradient descent with Adam optimization"
start_time = time.time()
para_adam,J_adam = Mini_Batch_GDAdam(x,y,layer_dim,n_epoch,batch_size,funhid,\
                                     funout,gradfun,LR,beta1,beta2,epsilon)
print("---Running GD with Adam optimization uses %s seconds ---" % (time.time() - start_time))


