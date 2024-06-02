# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:04:33 2024

@author: 44754
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################   

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
    #shape of z
    a,b = z.shape
    res = np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            res[i,j] = max(0,z[i,j])
            del j
        del i
    return res
        

def Leaky_ReLu(z):
    "compute the Leaky Relu fun"
    #shape of z
    a,b = z.shape
    res = np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            res[i,j] = max(0.01*z[i,j],z[i,j])
            del j
        del i
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