# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import packages and funs
import numpy as np
import matplotlib.pyplot as plt
import scipy
#import files
from lr_utilis import *



def sigmoid(z):
    "since our problem is a binary classifier, we choose the Sigmoid function\
        as our activation function"
    g = 1./(1+np.exp(-z))
    return g


def gradCost(para,x,y,m,n):
    "compute the cost function and the gradient"
    "forward propagation"
    #abstract w, and b
    b = para[0].reshape(1,1)
    w = para[1:].reshape(n,1)
    #compute the input for the activation fun
    z = x @ w + b
    #compute the prediction
    h = sigmoid(z)
    #make sure the labels is in vector form
    y = y.reshape(m,1)
    "cost function"
    J = -1./m * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    "back propagation"
    #departure between prediction and labels
    d = h - y
    #compute the gradient
    db = 1./m * np.sum(d)
    dw = 1./m * (d.T @ x)
    grad = np.zeros(n+1)
    grad[0] = db
    grad[1:] = dw
    
    return J.reshape(1), grad

def prediction(x,m,n,theta):
    "use the updated parameter produce our hypothesis output"
    #abstract the bias and weights
    b = theta[0].reshape(1,1)
    w = theta[1:].reshape(n,1)
    #forward propagation
    #compute the input for the activation fun
    z = x @ w + b
    #compute the prediction
    h = sigmoid(z)
   
    for i in range(m):
        h[i,0] = 1 if h[i,0] > 0.5 else 0
        del i
   
    return h.flatten()


def Logistic_regression(filename):
    "use the logistic regression"
    #read data and count the data size
    x,y,m,n,size = readHDF(filename)
    #standarlize the data
    x /= 255.
    #initialize parameters
    theta = np.zeros(n+1)
    #find the best fit parameters
    res = scipy.optimize.minimize(gradCost,theta,args=(x,y,m,n),\
                            jac=True, method='TNC',options={'maxfun': 10**5})
    theta_a = res.x
    #use the parameters to predict with our hypothesis
    h = prediction(x,m,n,theta_a)
    #compute the averaged square-error between prediction and labels
    error = np.sum(1./m * np.power(h-y,2))
    
    return x,y,theta_a,error

    
