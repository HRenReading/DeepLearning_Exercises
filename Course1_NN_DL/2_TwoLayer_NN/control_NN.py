# -*- coding: utf-8 -*-
"""
Created on Sat May  4 15:08:22 2024

@author: 44754
"""
from utilis import *
from OneHL_NN import *

#########################################################################
"parameters for the experiments"
K = 1     #number of classes.
n_unit = 4     #number of hidden units.
#method of minimization
methods = 'Newton-CG'
#number of maximum iterations used to minimize the cost
option = {'maxiter':30} 
#########################################################################
"experiment with NN using single Hidden Layer"
#update the parameter
para,x,y,m,n = NN_1HiddenLayer(n_unit,K,methods,option)
#prediction using trained model
pred = prediction(x, m, para, n, n_unit, K)
#check accuracy
print('Accuracy of the single-hidden-layer Neural Network is',\
      np.sum((y==pred))/m *100, '%','using '+methods)
