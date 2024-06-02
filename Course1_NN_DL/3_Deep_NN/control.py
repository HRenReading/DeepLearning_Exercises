# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:52:18 2024

@author: 44754
"""

#import packages and funs
import matplotlib.pyplot as plt
#import files and all the functions in the files
from Load_data import *
from Parameters import *
from propagation import *
from DeepL_NN import *
from turnOffWarning import *

############################################################################
"Hyperparameters for Neural network"
#number of unit in a single hidden layer
n_unit = 100
#number of unit in output layer
K = 1
#method of minimization
method = 'CG'
#options for the minimization methods
option = {'maxiter': None}
#number of layer in DeepL NN
L = 6
##########################################################################
#load the data to ndarrays
x_train, ytrain, x_test, ytest, classes = load_data()
#find the size of the data and reshape the training data
xtrain,mtrain,ntrain = reshape(x_train)
#reshape the test data
xtest,mtest,ntest = reshape(x_test)
##########################################################################
"experiment with L-layer (L>2) Neural Network"
#time the experiments
start_time = time.time()
#train the MLP with the training set
para_a = MLP(xtrain, ytrain, mtrain, ntrain, n_unit, K, L, \
             ReLu, sigmoid, gradReLu, method, option)
#print the time of training the model
print("Time of runing a "+str(L)+"-Layer NN is: %s seconds"%int(time.time()-start_time))
#prediction on the training data using the updated parameters
ptrain = prediction(xtrain,mtrain,para_a,L,ReLu,sigmoid)
#prediction on the test data using the updated parameters
ptest = prediction(xtest,mtest,para_a,L,ReLu,sigmoid)
print('The accuracy of the '+str(L)+'-Layer NN for pictures classification on the'
      ' training set is:',np.sum(ptrain==ytrain)/mtrain*100,'% ')
print('The accuracy of the '+str(L)+'-Layer NN for pictures classification on the'
      ' test set is:',np.sum(ptest==ytest)/mtest*100,'% with '+str(n_unit)+' unit')

