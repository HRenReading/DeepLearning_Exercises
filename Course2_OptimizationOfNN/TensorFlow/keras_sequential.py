# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:28:55 2024

@author: 44754
"""

#turn off the warning from tensorflow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#import packages
import numpy as np
#import tensorflow
import tensorflow as tf
import keras
from keras import initializers
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import SGD, Adam

def Sequential_Function(layer_dim, xtrain, ytrain, hidfun, outfun, 
                    init_weight, init_bias, epoch, batch_size, 
                    callback, optimizer, loss_fun):
    """
    Multiclassifier DeepL Neural Network.
    
    Input -- layer_dim, input size (number of features) and the number of 
             units in each layer.
             xtrain, training data set features.
             ytrain, training data set labels.
             hidfun, activation function used in hidden layers.
             outfun, activation function used in output layer.
             ini_weight, method we use to initialize the weightw, w.
             ini_bias, method we use to initialize the bias term, b.
             epoch, number of epochs we use in neural network.
             batch_size, number of samples in each mini-batch.
             callback, for early stop in epochs.
    Output -- the trained model using the training data set.
    """
    # use keras API
    model = tf.keras.Sequential()
    #number of layers, including the input layer
    L = len(layer_dim)
    #model structure
    for i in range(1,L):
        if i == 1:
            model.add(Dense(layer_dim[i], input_dim = layer_dim[i-1],
                      activation = hidfun, kernel_initializer = init_weight,
                      bias_initializer = init_bias))
        elif i == L - 1:
             model.add(Dense(layer_dim[i], activation = outfun,
                             kernel_initializer = init_weight,
                             bias_initializer = init_bias))
        else:
            model.add(Dense(layer_dim[i], activation = hidfun,
                            kernel_initializer = init_weight, 
                            bias_initializer = init_bias))
    model.compile(optimizer = optimizer, loss=loss_fun,
                  metrics = ['accuracy'])
    # Convert labels to categorical one-hot encoding
    labels = keras.utils.to_categorical(ytrain, 
                                        num_classes = layer_dim[-1])
    #train our model use the training set
    model.fit(xtrain, labels, epochs = epoch, batch_size = batch_size,
              callbacks= callback)

    return model









def test(model, xtest, ytest,layer_dim,batch_size):
    """
    Use our trained model to test on the test data set.
    
    Input -- model, our trained model.
             xtest, test data set features.
             ytest, test data set labels.
    """
    # Convert labels to categorical one-hot encoding (binary)
    labels = keras.utils.to_categorical(ytest, 
                                        num_classes = layer_dim[-1])
    #test data results use our trained model (loss and accuracy)
    results = model.evaluate(xtest, labels, batch_size = batch_size)
    print("Loss of our trained model on test data -- ",results[0])
    print("Accuracy our trained model on of test data -- ",results[1]*\
          100,'%')
    
    