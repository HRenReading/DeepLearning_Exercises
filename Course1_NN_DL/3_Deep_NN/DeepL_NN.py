# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:50:44 2024

@author: 44754
"""
import numpy as np
from scipy.optimize import minimize
from propagation import *
from Activa_fun import *
from Parameters import *
###########################################################################
"turn off the runtime warning"
import warnings
warnings.filterwarnings("ignore")
###########################################################################


def DL_costgrad(para_roll,x,y,m,n,L,n_unit,K,funhid,funout,gradfun):
    """
    Compute the cost and its gradient based on the current parameters.
    
    Input -- paramters, the dictionary current sets of parameters for all 
             layers.
             x, the training data.
             fun_hid, the activation fun of the hidden layer units.
             fun_out, the activation fun of the output layer.
    Output -- J, the cost. grad, the gradient of all sets of parameters.
    """
    #unpack parameters from array to the original formation
    parameters = unroll(para_roll,L,n,n_unit,K)
    #compute the output of the forward propagation
    output = forward_prop(parameters,x,L,funhid,funout)
    #final output from the forward model
    h = output['a'+str(L)]
    #cost
    J = -1./m * (np.dot(y,np.log(h.T)) + np.dot(1 - y,np.log(1 - h).T))
    #gradient of the cost
    grad = backward(parameters,output,gradfun,L,x,y,m)
    #roll gradient to a array
    grad = roll_grad(grad,L)

    return J.flatten(),grad

def MLP(x, y, m, n, n_unit, K, L, funfid, funout, gradfun, method, option):
    """
    Multi-layer perceptron DL.
    
    Input --
    Output -- 
    """
    #initialize the parameters for all layers
    parameters = iniPara(n_unit, n, K, L)
    #roll the parameters into an array
    para_roll = roll(parameters, L)
    #cost and its gradient
    J,grad = DL_costgrad(para_roll, x, y, m, n, L, n_unit, K, funfid,\
                         funout, gradfun)
    #minimize the cost function and its gradient   
    res = minimize(DL_costgrad,para_roll,args=(x, y, m, n, L, n_unit, K, \
                                               funfid, funout, gradfun),\
                   jac=True,method=method,options=option)
    #abstract the updated parameter from the result
    para_a = res.x
    #reform the rolled parameters to its original form
    para_a =unroll(para_a, L, n, n_unit, K)
    
    print("The model is using the ",method," method")
    return para_a
    
    
    

def prediction(x, m, para, L, funhid, funout):
    "compute the predition using updated parameters"
    #set a0 = x
    a = np.copy(x)
    for i in range(1,L+1):
        #compute the hidden layer
        if i != L:
            z = np.dot(para['w'+str(i)],a) + para['b'+str(i)]
            a = funhid(z)
        #compute the output layer
        else:
            z = np.dot(para['w'+str(i)],a) + para['b'+str(i)]
            a = funout(z)
        del i
    for i in range(m):
        a[:,i] = 1 if a[:,i] >= 0.5 else 0 
        del i
        
    return a


