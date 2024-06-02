# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:01:28 2024

@author: 44754
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
from scipy.io import loadmat

###########################################################################

def load_2D_dataset():
    data = loadmat('data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
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
    res = np.int64(z > 0)
        
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

def initialization(layer_dim,name_of_method):
    "initialize the weight with different method"
    np.random.seed(3) 
    #create an empty dictionary for parameters
    para = {}
    #count the number of levels
    L = len(layer_dim)
    #for the case we want zero weights
    if name_of_method == 'zeros':
        for i in range(1,L):
            para['b'+str(i)] = np.zeros((layer_dim[i],1))
            para['w'+str(i)] = np.zeros((layer_dim[i],layer_dim[i-1]))
            del i
    #for the case we want large random values for the weights
    elif name_of_method == 'random':
        for i in range(1,L):
            para['b'+str(i)] = np.zeros((layer_dim[i],1))
            para['w'+str(i)] = np.random.randn(layer_dim[i],\
                                                layer_dim[i-1])*10
            del i
    #for the case we want He Initialization
    elif name_of_method == 'he':
        for i in range(1,L):
            para['b'+str(i)] = np.zeros((layer_dim[i],1))
            para['w'+str(i)] = np.random.randn(layer_dim[i],layer_dim[i-1])\
                * np.sqrt(2./layer_dim[i-1])
            del i
            
    return para
 
###########################################################################
def forward_propagation_with_dropout_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(3, 5)
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    
    return X_assess, parameters


def backward_propagation_with_dropout_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(3, 5)
    Y_assess = np.array([[1, 1, 0, 1, 0]])
    cache = (np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
           [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]]), np.array([[ True, False,  True,  True,  True],
           [ True,  True,  True,  True, False]], dtype=bool), np.array([[ 0.        ,  0.        ,  4.27989081,  5.21401307,  0.        ],
           [ 0.        ,  8.32019881,  1.58102041,  2.92987024,  0.        ]]), np.array([[-1.09989127, -0.17242821, -0.87785842],
           [ 0.04221375,  0.58281521, -1.10061918]]), np.array([[ 1.14472371],
           [ 0.90159072]]), np.array([[ 0.53035547,  8.02565606,  4.10524802,  5.78975856,  0.53035547],
           [-0.69166075, -1.71413186, -3.81223329, -4.61667916, -0.69166075],
           [-0.39675353, -2.62563561, -4.82528105, -6.0607449 , -0.39675353]]), np.array([[ True, False,  True, False,  True],
           [False,  True, False,  True,  True],
           [False, False,  True, False, False]], dtype=bool), np.array([[ 1.06071093,  0.        ,  8.21049603,  0.        ,  1.06071093],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]), np.array([[ 0.50249434,  0.90085595],
           [-0.68372786, -0.12289023],
           [-0.93576943, -0.26788808]]), np.array([[ 0.53035547],
           [-0.69166075],
           [-0.39675353]]), np.array([[-0.7415562 , -0.0126646 , -5.65469333, -0.0126646 , -0.7415562 ]]), np.array([[ 0.32266394,  0.49683389,  0.00348883,  0.49683389,  0.32266394]]), np.array([[-0.6871727 , -0.84520564, -0.67124613]]), np.array([[-0.0126646]]))


    return X_assess, Y_assess, cache
    