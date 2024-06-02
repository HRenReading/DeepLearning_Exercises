# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:57:55 2024

@author: 44754
"""

from keras_sequential import *
from utilis import *

##########################################################################
#import data from the files
x_train,y_train,x_test,y_test,classes = load_dataset()
#standardlize the images data
xtrain, xtest = x_train/255., x_test/255.
ytrain,ytest = y_train.astype(int),y_test.astype(int)
#convert data to 2D (m,n)
xtrain,mtrain,ntrain = convert(xtrain)
xtest,mtest,ntest = convert(xtest)
"Hyperparameters"
batch_size = 32
#structure of the model (units in each layer including the input)
layer_dim = [ntrain,100,80,60,40,20,classes.size]
#activation fun used in hidden layers
hidfun = 'relu'
#activation fun used in output layer
outfun = 'softmax'
#optimizer stochastic gradient descent
opt1 = SGD(learning_rate = 1e-4, momentum=0.9)
##optimizer gradient descent with Adam algorithm
opt2 = Adam(learning_rate = 1e-4, beta_1 = 0.9, beta_2 = 0.999,
            epsilon = 1e-08,)
#initializer for the weights
init_weight = 'HeNormal'
#initializer for bias
init_bias = 'Zeros'
#activation fun used in hidden layers
hidfun = 'relu'
#activation function used in output layer
outfun = 'softmax'
#number of epochs used in NN
epoch = 100
#loss function
loss_fun = 'CategoricalCrossentropy'
#early stop for the neural network
callback = [keras.callbacks.EarlyStopping(monitor = "loss", patience = 5,
                                          mode = "min",
                                          restore_best_weights = True)]
###########################################################################
"""
"Experiment with picture adata set using Sequential function in keras"
#train the model with our 
model = Sequential_Function(layer_dim, xtrain, ytrain, hidfun, outfun, 
                        init_weight, init_bias, epoch, batch_size, 
                        callback, opt1, loss_fun)
#test our trained model on test data set
test(model, xtest, ytest,layer_dim,batch_size)
"""

