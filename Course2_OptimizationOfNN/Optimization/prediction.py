# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:40:28 2024

@author: 44754
"""

import numpy as np


def prediction(x,para,layer_dim,funhid,funout):
    "use updated parameters to predict the outcome"
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
    h = out['a'+str(L-1)]
    for i in range(x.shape[1]):
        h[0,i] = 1 if h[0,i] >= 0.5 else 0
        del i
          
    return h

def accuracy(p,y,method_name):
    "print the acuracy of trained model on data set"
    m = y.shape[1]
    print('NN accuracy',method_name,np.sum(p==y)/m*100,'% ')
    
    