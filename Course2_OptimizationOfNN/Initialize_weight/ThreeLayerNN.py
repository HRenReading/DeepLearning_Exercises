# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:00:43 2024

@author: 44754
"""

#import packages and functions
import numpy as np
#import files and functions
from utilis import *
from Initialize_weight import *

###########################################################################
"forward and back propagation functions"
def forward(x,para,layer_dim,funhid,funout):
    "compute the forward propagation"
    #count the number of layers
    L = len(layer_dim)
    #empty dictionary for all layers output
    out = {}
    #forward propagation
    for i in range(1,L):
        if i == 1:
            out['z'+str(i)] = np.dot(para['w'+str(i)],x) + para['b'+str(i)]
            out['a'+str(i)] = funhid(out['z'+str(i)])
        elif i == L-1:
            out['z'+str(i)] = np.dot(para['w'+str(i)],out['a'+str(i-1)]) + \
                              para['b'+str(i)]
            out['a'+str(i)] = funout(out['z'+str(i)])
        else:
            out['z'+str(i)] = np.dot(para['w'+str(i)],out['a'+str(i-1)]) + \
                              para['b'+str(i)]
            out['a'+str(i)] = funhid(out['z'+str(i)])
            
    return out

def cost(h,y,m):
    """
    Compute the cost.

    Input -- h, output from neural network.
             y, labels from training set
             m, size of training set.
    Output -- cost fun result.
    """
    J = -1./m * (np.dot(y,np.log(h.T)) + np.dot(1 - y,np.log(1 - h.T)))
    
    return J
    

def backward(x,y,m,n,out,para,layer_dim,gradfun):
    "compute the back propagation and store the gradient of the parameters"
    #count the number of layers in the NN
    L = len(layer_dim)    
    #empty dictionary for the gradient of bias and weights
    grad = {}
    for i in range(L-1,0,-1):
        if i == L-1:
            grad['dz'+str(i)] = 1./m * (out['a'+str(i)] - y)
            grad['dw'+str(i)] = np.dot(grad['dz'+str(i)],out['a'+str(i-1)].T)
        elif i == 1:
            grad['dz'+str(i)] = np.multiply(np.dot(para['w'+str(i+1)].T,\
                        grad['dz'+str(i+1)]),gradfun(out['z'+str(i)]))
            grad['dw'+str(i)] = np.dot(grad['dz'+str(i)],x.T)
        else:
            grad['dz'+str(i)] = np.multiply(np.dot(para['w'+str(i+1)].T,\
                        grad['dz'+str(i+1)]),gradfun(out['z'+str(i)]))
            grad['dw'+str(i)] = np.dot(grad['dz'+str(i)],out['a'+str(i-1)].T)
        
        grad['db'+str(i)] = np.sum(grad['dz'+str(i)],axis=1,\
                                          keepdims = True)
        del i
            
    return grad

def gradient_descent(para,grad,LR):
    "compute the updated parameters using gradient descent"
    #number of parameters
    npara = int(len(para)/2)
    #update all weights and biases
    for k in range(npara):
        para["w"+str(k+1)] = para["w"+str(k+1)]-LR*grad["dw" + str(k+1)]
        para["b"+str(k+1)] = para["b"+str(k+1)]-LR*grad["db" + str(k+1)]
        del k
    return para

def DeepL_NN(x,y,layer_dim,funhid,funout,gradfun,niter,LR,ini_name):
    """
    Update the parameters using a L-layer neural network.
    
    Input -- x, training data features.
             y, training data labels.
             layer_dim, number of units in each layer (inluding the input).
             funhid, activation fun used in the hidden layers.
             funout, activation fun used in the output layer.
             gradfun, the gradient fun of the hidden layer activation fun.
             niter, number of iteration to compute the gradient descent.
             LR, learning rate of gradient descent.
    Output -- para, updated parameters. J, the cost after each iterations.
    """
    #find the number of features and sample size
    n,m = x.shape
    #initialize the weights and bias
    para = initialization(layer_dim,ini_name)
    #number of sets of parameters
    npara = len(layer_dim) - 1
    #empty array for all cost in every iteration
    J = np.zeros(niter+1)
    #loop over niter iterations
    for i in range(niter+1):
        #forward propagation
        out = forward(x,para,layer_dim,funhid,funout)
        #compute the cost
        J[i] = cost(out['a'+str(npara)],y,m)[0,0]
        #compute the gradient of parameters
        grad = backward(x,y,m,n,out,para,layer_dim,gradfun)
        #update the parameters using gradient descent
        para = gradient_descent(para,grad,LR)
        del i
    
    return para,J
        
###########################################################################

def prediction(para,x,layer_dim,funhid,funout):
    "using updated parameters to predict the outcome"
    p = forward(x,para,layer_dim,funhid,funout)['a'+str(len(layer_dim)-1)]
    for i in range(x.shape[1]):
        p[0,i] = 1 if p[0,i] >= 0.5 else 0
        del i
    return p

def accuracy(p,y,data_name):
    "print the acuracy of trained model on data set"
    m = y.shape[1]
    print('The accuracy of the Neural Network on the',data_name,'set is:',np.sum(p==y)/m*100,'% ')
    
    
    
            