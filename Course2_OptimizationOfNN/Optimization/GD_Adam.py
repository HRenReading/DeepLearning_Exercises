# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:14:49 2024

@author: 44754
"""

import numpy as np
from utilis import *
from prediction import *

###############################################################################
"forward and back propagation functions"
def forward(x,para,layer_dim,funhid,funout):
    """
    Forward propagation in neural network
    
    Input -- x, data features. 
             para, current parameters.
             layer_dim, the number of input units in each layer including
                        the input layer.
             funhid, activation fun used in hidden layers.
             funout, activation fun used in the output layer.
    """
    #count the number of layers
    L = len(layer_dim)
    #empty dictionary for all layers output
    out = {}
    #forward propagation
    for i in range(1,L):
        if i == 1:
            out['z'+str(i)] = np.dot(para['w'+str(i)],x) + para['b'+str(i)]
            out['a'+str(i)] = funhid(out['z'+str(i)])
        elif i != 1 and i != L-1:
            out['z'+str(i)] = np.dot(para['w'+str(i)],out['a'+str(i-1)])+para['b'+str(i)]
            out['a'+str(i)] = funhid(out['z'+str(i)])
        elif i == L-1:
            out['z'+str(i)] = np.dot(para['w'+str(i)],out['a'+str(i-1)])+para['b'+str(i)]
            out['a'+str(i)] = funout(out['z'+str(i)])         
            
    return out

def cost(h,y):
    """
    Compute the cost.

    Input -- h, final output from forward propagation.
             y, labels from training set
    Output -- cost fun result.
    """
    #count the number of examples
    m = len(y[0,:])
    #cost function without regularization
    J = -1./m * (np.dot(y,np.log(h).T) + np.dot(1 - y,np.log(1 - h).T))
        
    return J
    

def backward(x,y,out,para,layer_dim,gradfun):
    """
    Compute the gradient of bias and weights using back propagation.
    
    Input -- x, data features. 
             y, data labels.
             out, final output from forward propagation.
             para, current parameters.
             layer_dim, the number of input units in each layer including
                        the input layer.
             gradfun, the gradient of the activation fun in the hidden layers.
    Output -- the gradient of bias and weights.
    """
    #find the shape of data
    n,m = x.shape
    #count the number of layers in the NN
    L = len(layer_dim)    
    #empty dictionary for the gradient of bias and weights
    grad = {}
    #compute the gradiets
    for i in range(L-1,0,-1):
        #for the output layer
        if i == L-1:
            grad['dz'+str(i)] = out['a'+str(i)] - y
            grad['db'+str(i)] = np.sum(grad['dz'+str(i)],axis=1,keepdims = True)/m
            grad['dw'+str(i)] = np.dot(grad['dz'+str(i)],out['a'+str(i-1)].T)/m
        elif i != L-1 and i != 1:
            grad['da'+str(i)] = np.dot(para['w'+str(i+1)].T,grad['dz'+str(i+1)])
            grad['dz'+str(i)] = np.multiply(grad['da'+str(i)],gradfun(out['a'+str(i)]))
            grad['db'+str(i)] = np.sum(grad['dz'+str(i)],axis=1,keepdims = True)/m
            grad['dw'+str(i)] = np.dot(grad['dz'+str(i)],out['a'+str(i-1)].T)/m
        else:
            grad['da'+str(i)] = np.dot(para['w'+str(i+1)].T,grad['dz'+str(i+1)])
            grad['dz'+str(i)] = np.multiply(grad['da'+str(i)],gradfun(out['a'+str(i)]))
            grad['db'+str(i)] = np.sum(grad['dz'+str(i)],axis=1,keepdims = True)/m
            grad['dw'+str(i)] = np.dot(grad['dz'+str(i)],x.T)/m
        del i
            
    return grad

def GD_Adam(para,grad,v,s,beta1,beta2,LR,Adam_counter,epsilon):
    "Adam algorith with gradient descent"
    #number of parameters
    npara = int(len(para)/2)
    #update all weights and biases
    for k in range(npara):
        # compute velocities
        v["dw"+str(k+1)] = beta1*v["dw"+str(k+1)]+(1-beta1)*grad['dw'+str(k+1)]\
                                                       /(1-beta1**Adam_counter)
        v["db"+str(k+1)] = beta1*v["db"+str(k+1)]+(1-beta1)*grad['db'+str(k+1)]\
                                                       /(1-beta1**Adam_counter)
        #compute the RMSprop
        s["dw"+str(k+1)] = beta2*s["dw"+str(k+1)]+(1-beta2)*np.power(\
                                grad['dw'+str(k+1)],2)/(1-beta2**Adam_counter)
        s["db"+str(k+1)] = beta2*s["db"+str(k+1)]+(1-beta2)*np.power(\
                              grad['db'+str(k+1)],2)/(1 - beta2**Adam_counter)
        #update the parameters using Adam optimization
        para["w"+str(k+1)] = para["w"+str(k+1)]-LR*v["dw"+str(k+1)]/\
                             np.sqrt(s["dw"+str(k+1)]+epsilon)                           
        para["b"+str(k+1)] = para["b"+str(k+1)]-LR*v["db"+str(k+1)]/\
                             np.sqrt(s["db"+str(k+1)]+epsilon)                       
        del k
    
    return para,v,s

def Mini_Batch_GDAdam(x,y,layer_dim,n_epoch,batch_size,funhid,funout,gradfun,LR\
                      ,beta1,beta2,epsilon):
    "Neural network using gradient descent with Adam optimization"
    #initialize parameters
    para = initialization(layer_dim) 
    #initialize velocity for momentum and RMSprop
    v,s = initial_Adam(layer_dim)
    #initialize cost
    J = np.zeros(n_epoch+1)
    #count the number of layers in NN
    L = len(layer_dim)-1
    #loop over n_epoch
    seed = 10
    #initialize adam counter
    t = 0
    for i in range(n_epoch+1):
        #create the mini batches with shuffled training set
        mini_batch = shuffle(x,y,batch_size,seed = seed+1)
        #count the number of mini-batches
        n_batch = int(len(mini_batch)/2)
        for j in range(n_batch):
            #forward propagation for single mini-batch
            out = forward(mini_batch['X'+str(j+1)],para,layer_dim,funhid,funout)
            #compute the cost of current epoch
            J[i] = cost(out['a'+str(L)],mini_batch['Y'+str(j+1)])[0,0]
            #gradient of parameters with single batch
            grad = backward(mini_batch['X'+str(j+1)],mini_batch['Y'+str(j+1)],out,\
                            para,layer_dim,gradfun)
            t = t+1
            #update the parameters
            para,v,s = GD_Adam(para,grad,v,s,beta1,beta2,LR,t,epsilon)
            del j
        del i
    #compute the prediction
    p = prediction(x,para,layer_dim,funhid,funout)    
    #print out the accuracy when using GD
    accuracy(p,y,'Gradient descent with Adam optimization')  

    return para,J      


        

