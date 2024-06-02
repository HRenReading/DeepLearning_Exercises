# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:07:28 2024

@author: 44754
"""

import numpy as np
import math
import sklearn.datasets

###############################################################################
"generate random data set"
def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
    # Visualize the data
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return train_X, train_Y

###############################################################################
"activation functions and its gradient"
def sigmoid(z):
    "compute Sigmoid fun"
    g = 1./(1. + np.exp(-z))
    return g

def ReLu(z):
    "compute the Relu fun"
    res = np.maximum(0,z)
    return res

def gradReLu(z):
    "compute the gradient of ReLu funtion at z"
    res = np.int64(z > 0)
        
    return res

###############################################################################
"parameter initalization"
def initialization(layer_dim):
    "initialize the weight with He initialization method"
    #number of layers in the neural network
    L = len(layer_dim)
    #empty dictionary to store all parameters
    para  ={}
    np.random.seed(3) 
    for i in range(1,L):
        #initialize the bias for ith layer
        para['b'+str(i)] = np.zeros((layer_dim[i],1))
        #initialize the weights for ith layer
        para['w'+str(i)] = np.random.randn(layer_dim[i],layer_dim[i-1])\
                * np.sqrt(2./layer_dim[i-1])
        del i
            
    return para

def initial_velocity(layer_dim):
    "initialize the velocity for gradient descent with momentum"
    #number of layers in the neural network
    L = len(layer_dim)
    #empty dictionary to store all parameters
    v  ={}
    for i in range(1,L):
        #initialize the bias for ith layer
        v['db'+str(i)] = np.zeros((layer_dim[i],1))
        #initialize the weights for ith layer
        v['dw'+str(i)] = np.zeros((layer_dim[i],layer_dim[i-1]))
        del i

    return v

def initial_Adam(layer_dim):
    "initialize the velocity for gradient descent with momentum"
    #number of layers in the neural network
    L = len(layer_dim)
    #empty dictionary to store all parameters
    v  ={}
    s = {}
    for i in range(1,L):
        #initialize the bias for ith layer
        v['db'+str(i)] = np.zeros((layer_dim[i],1))
        s['db'+str(i)] = np.copy(v['db'+str(i)])
        #initialize the weights for ith layer
        v['dw'+str(i)] = np.zeros((layer_dim[i],layer_dim[i-1]))
        s['dw'+str(i)] = np.copy(v['dw'+str(i)])
        del i

    return v,s

###############################################################################
"shuffle the orginal data to create the mini-Batch"
def shuffle(x,y,batch_size,seed):
    np.random.seed(seed)
    #the numbers of features and samples in the data
    n,m = x.shape
    #shaffle data
    shuffled_index = np.random.permutation(m)
    x = x[:,shuffled_index]
    y = y[:,shuffled_index]
    #count the number of mini-Batches we have using round down method
    n_batch = math.floor(m/batch_size)
    #empty dictionary for all mini-batches
    mini_batch = {}
    #partition all shuffled data into mini-batches
    for i in range(n_batch):
        mini_batch['X'+str(i+1)] = x[:,i*batch_size:(i+1)*batch_size]
        mini_batch['Y'+str(i+1)] = y[:,i*batch_size:(i+1)*batch_size]
        del i
    #in the case we have some leftover samples smaller than the mini-batch size
    if m % batch_size != 0:
        mini_batch['X'+str(n_batch+1)] = x[:,n_batch*batch_size:]
        mini_batch['Y'+str(n_batch+1)] = y[:,n_batch*batch_size:]
    
    return mini_batch
    
    



