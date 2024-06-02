# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:19:45 2024

@author: 44754
"""
#import packages and funs
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#import files
from utilis import *
np.random.seed(1)


def forward(w1,b1,w2,b2,x):
    "compute the forward propagation using current parameters"
    #hidden layer input
    z1 = np.dot(w1,x) + b1
    #hidden layer output
    a1 = tanh(z1)
    #output layer input
    z2 = np.dot(w2,a1) + b2
    #final output
    a2 = sigmoid(z2)
    return z1,a1,z2,a2

def backward(z1,z2,a1,a2,x,y,m,w2,b2):
    #gradient of output layer
    dz2 = a2 - y
    #gradient of the second bias term and weights
    db2 = np.sum(dz2)/m
    dw2 = np.dot(dz2,a1.T)/m
    #gradient of hidden layer
    dz1 = np.multiply(np.dot(w2.T,dz2),gradTanh(z1))
    #gradient of the first bias term and weights
    db1 = np.sum(dz1,axis=1)/m
    dw1 = np.dot(dz1,x.T)/m
    
    return db1,dw1,db2,dw2

def cost_grad(theta_roll,x,y,m,n,n_unit,K):
    "compute the cost and its gradient"
    #unpack the 2 sets of parameters
    w1,b1,w2,b2 = unroll_para(theta_roll, n, n_unit, K)
    #forward propagation
    z1,a1,z2,a2 = forward(w1,b1,w2,b2,x)    
    #compute the cost
    J = -1./m * (np.dot(y,np.log(a2.T)) + np.dot(1 - y,np.log(1 - a2.T)))
    #back propagation
    db1,dw1,db2,dw2 = backward(z1,z2,a1,a2,x,y,m,w2,b2)
    droll = roll_para(dw1, db1, dw2, db2)
    
    return J.flatten(),droll

def NN_1HiddenLayer(n_unit,K,methods,option):
    "combine all functions together to form a Neural Network with 1 HL"
    #generate data (x,y) features and labels.
    x,y = load_planar_dataset()
    #count numbers of samples and features.
    m,n = x.T.shape    
    #initilize the parameters.
    w1,b1,w2,b2 = initialize(n,n_unit,K)    
    #roll all parameters into an array.
    theta_roll = roll_para(w1, b1, w2, b2)   
    res = minimize(cost_grad,theta_roll,args=(x,y,m,n,n_unit,K),jac=True,\
                   method=methods,options=option)
    para_roll = res.x
    
    return para_roll,x,y,m,n

def prediction(x, m, para_roll, n, n_unit, K):
    "compute the predition using updated parameters"
    #unpack parameters
    w1a,b1a,w2a,b2a = unroll_para(para_roll, n, n_unit, K)
    #make prediction and transfer it to binary form
    p = forward(w1a, b1a, w2a, b2a, x)[-1]
    for i in range(m):
        p[0,i] = 1 if p[0,i] > 0.5 else 0
        del i
    return p
    


