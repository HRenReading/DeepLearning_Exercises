# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:49:05 2024

@author: 44754
"""

import numpy as np
import sklearn.datasets

###########################################################################

"generate the training & test data sets"
def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

###########################################################################

"activation functions"
def sigmoid(z):
    "compute Sigmoid fun"
    g = 1./(1. + np.exp(-z))
    return g

def tanh(z):
    "compute the tanh fun"
    g = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
    return g

def ReLu(z):
    "compute the Relu fun"
    res = np.maximum(0,z)
    return res
        

def Leaky_ReLu(z):
    "compute the Leaky Relu fun"
    res = np.maximum(0.01*z,z)
    return res

###########################################################################

"gradient of the cost functions"
def gradSig(z):
    "compute the gradient of Sigmoid function at z"
    g = sigmoid(z) * (1 - sigmoid(z))
    return g

def gradTanh(z):
    "compute the gradient of tanh function at z"
    g = 1 - np.power(tanh(z))
    return g

def gradReLu(z):
    "compute the gradient of ReLu funtion at z"
    #shape of z
    a,b = z.shape
    res = np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            res[i,j] = 0 if z[i,j] < 0 else 1
            del j
        del i
        
    return res

def gradLeaky(z):
    "compute the gradient of Leaky-ReLu funtion at z"
    #shape of z
    a,b = z.shape
    res = np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            res[i,j] = 0.01 if z[i,j] < 0 else 1
            del j
        del i
        
    return res

###########################################################################