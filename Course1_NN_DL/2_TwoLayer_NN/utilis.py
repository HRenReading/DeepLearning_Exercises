# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:17:45 2024

@author: 44754
"""

import numpy as np
#set consistent random process
np.random.seed(1)

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y


def sigmoid(z):
    "sigmoid activation function for the output layer"
    g = 1./(1. + np.exp(-z))
    return g

def tanh(z):
    "tanh activation fun for the hidden layer"
    g = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
    return g

def gradTanh(z):
    "the function of gradient of the sigmoid function"
    grad = 1 - np.power(tanh(z),2)
    return grad

def roll_para(w1,b1,w2,b2):
    "roll the parameters into an array"
    #for first set of bias and weights
    theta1 = np.concatenate((b1,w1),axis=1)
    #for second set of bias and weights
    theta2 = np.concatenate((np.matrix(b2),w2),axis=1)
    #return an array contains all paras
    return np.concatenate((np.ravel(theta1,'F'),np.ravel(theta2,'F')))

def unroll_para(roll,n,n_unit,K):
    "unpack the 2 sets of parameters from the rolled array"
    #first set of parameters
    theta1 = roll[:(n+1)*n_unit].reshape(n+1,n_unit).T
    b1 = theta1[:,0]
    w1 = theta1[:,1:]
    #second set of parameters
    theta2 = roll[(n+1)*n_unit:].reshape(K,n_unit+1).T
    b2 = theta2[0,:]
    w2 = theta2[1:,:]
    
    return w1,np.matrix(b1).T,w2.T,np.matrix(b2)

def initialize(n,n_unit,K):
    "initialize the 2 sets of bias and weights"
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    #initialize the bias
    b1 = np.zeros([n_unit,1])
    b2 = np.zeros([K,1])
    #initialize weights
    W1 = np.random.randn(n_unit,n) * 0.01
    W2 = np.random.randn(K,n_unit) * 0.01
    
    return W1,b1,W2,b2

