# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:24:21 2024

@author: 44754
"""

import numpy as np

###########################################################################

def forward_prop(parameters,x,L,fun_hid,fun_out):
    """
    Compute the forward propagation using current parameters.
    
    Input -- paramters, the dictionary current sets of parameters for all 
             layers.
             x, the training data.
             fun_hid, the activation fun of the hidden layer units.
             fun_out, the activation fun of the output layer.
    Output -- A dictionary of all the output from all layers.
    """
    #create a dictionary to store all the output
    output = {}
    #compute z (input) and a (output) for all layers
    for i in range(1,L+1):
        #for the first hidden layer
        if i == 1:
            output['z'+str(i)] = np.dot(parameters['w'+str(i)],x) + \
                parameters['b'+str(i)]
            output['a'+str(i)] = fun_hid(output['z'+str(i)])
        #for the output layer
        elif i == L:
            output['z'+str(i)] = np.dot(parameters['w'+str(i)],\
                output['a'+str(i-1)]) + parameters['b'+str(i)]
            output['a'+str(i)] = fun_out(output['z'+str(i)])
        #for the other hidden layers
        else:
            output['z'+str(i)] = np.dot(parameters['w'+str(i)],\
                output['a'+str(i-1)]) + parameters['b'+str(i)]
            output['a'+str(i)] = fun_hid(output['z'+str(i)])
            
    return output

def backward(parameters,output,gradfun,L,x,y,m):
    """
    Compute the back propagation based on the current parameters and forward
    output.
    
    Input -- parameters, the dictionary of all sets of current parameters
             of all levels.
             output, the output from the forward propagagtion.
             gradfun, the derivative of the activation function.
    Output -- gradient of all sets of bias and weights.
    """
    #empty dictionary for storing the gradients of the biases and weights
    grad = {}
    #empty dictionary for storing the gradients of output of all layers
    da = {}
    for i in range(L,0,-1):
        #for the last set of bias and weights
        if i == L:
            da['da'+str(L)] = output['a'+str(L)] - y
            grad['dw'+str(L)] = \
                np.dot(da['da'+str(L)],output['a'+str(L-1)].T)/m 
        #for the first set of bias and weights
        elif i == 1:
            da['da'+str(1)] = np.multiply(np.dot(parameters['w'+str(2)].T,\
                da['da'+str(2)]),gradfun(output['z'+str(1)]))
            grad['dw'+str(1)] = np.dot(da['da'+str(1)],x.T)/m 
        else:
            da['da'+str(i)] = np.multiply(np.dot(\
                parameters['w'+str(i+1)].T,da['da'+str(i+1)]),\
                        gradfun(output['z'+str(i)]))
            grad['dw'+str(i)] = np.dot(da['da'+str(i)],\
                                      output['a'+str(i-1)].T)/m
           
        grad['db'+str(i)] = np.matrix(np.sum(da['da'+str(i)],axis=1)/m)
    
    return grad
    
    
    