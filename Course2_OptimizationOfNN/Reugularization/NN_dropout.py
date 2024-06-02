# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:49:17 2024

@author: 44754
"""

import numpy as np
from utilis import *

"forward and back propagation functions"
def forward_drop(x,para,layer_dim,funhid,funout,keep_prob):
    "compute the forward propagation"
    np.random.seed(1)
    #count the number of layers
    L = len(layer_dim)
    #empty dictionary for all layers output
    out = {}
    #forward propagation
    for i in range(1,L):
        if i == 1:
            out['z'+str(i)] = np.dot(para['w'+str(i)],x) + para['b'+str(i)]
            out['a'+str(i)] = funhid(out['z'+str(i)])
            D1 = np.random.rand(out['a'+str(i)].shape[0],out['a'+str(i)].shape[1])     
            out['D1'] = D1 < keep_prob
            out['a'+str(i)] = out['a'+str(i)] * out['D1']
            out['a'+str(i)] /= keep_prob
        elif i != 1 and i != L-1:
            out['z'+str(i)] = np.dot(para['w'+str(i)],out['a'+str(i-1)]) + \
                              para['b'+str(i)]
            out['a'+str(i)] = funhid(out['z'+str(i)])
            D = np.random.rand(out['a'+str(i)].shape[0],out['a'+str(i)].shape[1])  
            out['D'+str(i)] = D < keep_prob
            out['a'+str(i)] = out['a'+str(i)] * out['D'+str(i)]
            out['a'+str(i)] /= keep_prob
        elif i == L-1:
            out['z'+str(i)] = np.dot(para['w'+str(i)],out['a'+str(i-1)]) + \
                              para['b'+str(i)]
            out['a'+str(i)] = funout(out['z'+str(i)])         
         
    return out


def backward_drop(x,y,out,para,layer_dim,gradfun,keep_prob):
    """compute the back propagation and store the gradient of the 
    parameters"""
    #count the number of layers in the NN
    L = len(layer_dim)    
    #find the size of x
    n,m = x.shape
    #empty dictionary for the gradient of bias and weights
    grad = {}
    for i in range(L-1,0,-1):
        if i == L-1:
            grad['dz'+str(i)] = out['a'+str(i)] - y
            grad['db'+str(i)] = 1./m * np.sum(grad['dz'+str(i)],axis=1,keepdims = True)
            grad['dw'+str(i)] = 1./m * (np.dot(grad['dz'+str(i)],out['a'+str(i-1)].T))
        elif i != 1 and i != L-1:
            grad['da'+str(i)] = np.dot(para['w'+str(i+1)].T,grad['dz'+str(i+1)])
            grad['da'+str(i)] = grad['da'+str(i)]*out['D'+str(i)]/keep_prob
            grad['dz'+str(i)] = np.multiply(grad['da'+str(i)],gradfun(out['a'+str(i)]))
            grad['db'+str(i)] = 1./m * np.sum(grad['dz'+str(i)],axis=1,keepdims = True)
            grad['dw'+str(i)] = 1./m * np.dot(grad['dz'+str(i)],out['a'+str(i-1)].T)
        else:
            grad['da'+str(i)] = np.dot(para['w'+str(i+1)].T,grad['dz'+str(i+1)])
            grad['da'+str(i)] = np.multiply(grad['da'+str(i)],out['D'+str(i)])/keep_prob
            grad['dz'+str(i)] = np.multiply(grad['da'+str(i)],gradfun(out['a'+str(i)]))
            grad['db'+str(i)] = 1./m * np.sum(grad['dz'+str(i)],axis=1,keepdims = True)
            grad['dw'+str(i)] = 1./m * np.dot(grad['dz'+str(i)],x.T)
            
        del i
            
    return grad

def cost(h,y,para):
    """
    Compute the cost.

    Input -- h, output from neural network.
             y, labels from training set
             m, size of training set.
             reg_name, the name of regularization method we use.
    Output -- cost fun result.
    """
    #count the number of examples
    m = len(y[0,:])
    #cost function without regularization
    J = -1./m * (np.dot(y,np.log(h).T) + np.dot(1 - y,np.log(1 - h).T))
    return J

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

def NN_drop(x,y,layer_dim,funhid,funout,gradfun,niter,LR,keep_prob):
    "update the parameters using gradient descent with dropout regularization"
    "in the neural network"
    #initial para
    para = initialization(layer_dim,'he')
    #number of sets of parameters
    npara = len(layer_dim) - 1
    #empty array for all cost in every iteration
    J = np.zeros(niter+1)
    #loop over niter iterations
    #loop over niter iterations
    for i in range(niter+1):
        #forward propagation
        out = forward_drop(x,para,layer_dim,funhid,funout,keep_prob)
        #compute the cost
        J[i] = cost(out['a'+str(npara)],y,para)[0,0]
        #compute the gradient of parameters
        grad = backward_drop(x,y,out,para,layer_dim,gradfun,keep_prob)
        #update the parameters using gradient descent
        para = gradient_descent(para,grad,LR)
        del i
        
    return para,J
        