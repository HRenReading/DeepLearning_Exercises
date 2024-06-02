# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:46:45 2024

@author: 44754
"""

import numpy as np

def iniPara(n_unit,n,K,L):
    """
    Initialize the parameters for all layers.
    
    Input -- n_unit, number of units in each hidden layer.
             n, number of features of the data set. 
             K, number of output unit.
             L, number of layers in the neural network.
    Output -- A dictionary of all sets of parameters for all layers.
    """
    np.random.seed(1)
    #create a dictionary for all sets of w and b
    parameters = {}
    #create the parameters for multiple layers
    for i in range(1,L+1):
        if i == 1:
            parameters['w' + str(i)] = np.random.randn(n_unit,n) * 0.01
            parameters['b' + str(i)] = np.zeros((n_unit,1))
        elif i == L:
            parameters['w' + str(i)] = np.random.randn(K,n_unit) * 0.01
            parameters['b' + str(i)] = np.zeros((K,1))
        else:
            parameters['w' + str(i)] = np.random.randn(n_unit,n_unit) * 0.01
            parameters['b' + str(i)] = np.zeros((n_unit,1))
     
    return parameters

def roll(parameters,L):
    "reshape the parameters for minimize the cost function"
    theta = {}
    #concatenate each set of bias and weights
    for i in range(1,L+1):
        w,b = parameters['w'+str(i)],parameters['b'+str(i)]
        theta['para'+str(i)] = np.concatenate((b,w),axis=1)
        del i
    #roll each set of parameters into an array
    para1,para2 = theta['para'+str(1)],theta['para'+str(2)]
    roll = np.concatenate((np.ravel(para1,'F'),np.ravel(para2,'F')))
    for i in range(2,L):
        para = theta['para'+str(i+1)]
        roll = np.concatenate((roll,np.ravel(para,'F')))
        del i
        
    return roll

def roll_grad(grad,L):
    "reshape the parameters for minimize the cost function"
    dtheta = {}
    #concatenate each set of bias and weights
    for i in range(1,L+1):
        dw,db = grad['dw'+str(i)],grad['db'+str(i)]
        dtheta['dpara'+str(i)] = np.concatenate((db,dw),axis=1)
        del i
    #roll each set of parameters into an array
    dpara1,dpara2 = dtheta['dpara'+str(1)],dtheta['dpara'+str(2)]
    droll = np.concatenate((np.ravel(dpara1,'F'),np.ravel(dpara2,'F')))
    for i in range(2,L):
        dpara = dtheta['dpara'+str(i+1)]
        droll = np.concatenate((droll,np.ravel(dpara,'F')))
        del i
        
    return droll
    
def unroll(para_roll,L,n,n_unit,K):
    "unpack the parameters array to its original form"
    #empty dictionary for storing all sets of parameters
    theta = {}
    #empty dictionary for stroring all bias terms and weights
    parameters = {}
    #from array to concatenated parameters
    for i in range(1,L+1):
        if i == 1:
            theta['para1'] = para_roll[:n_unit*(n+1)].reshape(n+1,n_unit).T
            parameters['b1'] = np.matrix(theta['para1'][:,0]).T
            parameters['w1'] = theta['para1'][:,1:]
        elif i == L:
            theta['para'+str(L)] = \
                    para_roll[n_unit*(n+1)+(L-2)*(n_unit*(n_unit+1)):].\
                        reshape(n_unit+1,K).T
            parameters['b'+str(L)] = np.matrix(theta['para'+str(L)][:,0]).T
            parameters['w'+str(L)] = theta['para'+str(L)][:,1:]
        else:
            theta['para'+str(i)] = \
                para_roll[n_unit*(n+1)+(i-2)*(n_unit*(n_unit+1)):\
                n_unit*(n+1)+(i-1)*(n_unit*(n_unit+1))].reshape(n_unit+1,n_unit).T
            parameters['b'+str(i)] = np.matrix(theta['para'+str(i)][:,0]).T
            parameters['w'+str(i)] = theta['para'+str(i)][:,1:]
                    
        del i
            
    return parameters
        
