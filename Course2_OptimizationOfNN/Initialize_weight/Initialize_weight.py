# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:40:55 2024

@author: 44754
"""

import numpy as np


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
                *np.sqrt(2./layer_dim[i-1])
            del i
            
    return para

    


