# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:27:31 2024

@author: 44754
"""
#import APIs
from tensorflow.keras import layers, models

def Model(n_f,f,size,maxp,s_conv,conv_fun,K):
    """
    Build the CNN using tensorflow.keras

    Input -- n_f, a list of number of filters in each layer.
             f, a list of height/widith of filters in each layer.
             size, image size (pix,pix,color_channels).
             maxp, list of window sizes of the maxpool in each layer.
             s_conv, stride size in convolution layer.
             s_max, stride size in maxpool layer
             conv_fun, activation function used in convolution layers.
    Output -- CNN model.
    """
    #initialize a model with Sequential function
    model = models.Sequential()
    #number of convolution/maxpool layers
    L = len(n_f)
    #add convolution and maxpool layers
    for i in range(L):
        if i == 0:
            model.add(layers.Conv2D(n_f[i], (f[i], f[i]), 
                                    activation = conv_fun,
                                    input_shape = size, padding='same',
                                    strides=(s_conv[i], s_conv[i]), 
                                    kernel_initializer='he_normal',
                                    bias_initializer='zeros'))
            
        else:
            model.add(layers.Conv2D(n_f[i], (f[i], f[i]), 
                                    activation = conv_fun,
                                    padding='same',
                                    strides=(s_conv[i], s_conv[i]), 
                                    kernel_initializer='he_normal',
                                    bias_initializer='zeros'))
        
        model.add(layers.MaxPooling2D(pool_size = (maxp[i], maxp[i]),
                                      padding = 'valid',
                                      strides = (maxp[i], maxp[i])))  
        del i
    #add the Dense layer for the output
    model.add(layers.Flatten())
    #model.add(layers.Dense(64, activation='softmax'))
    model.add(layers.Dense(K, activation = None))
    
    
    return model

def CNN(model, opt, xtrain, ytrain, xtest, ytest, loss_fun, batch_size, 
        n_epoch, callback, metrics):
    """
    Use the CNN model we built to fit the training set and evaluate on test data.
    
    Input -- model, the CNN we built.
             opt, optimization algorithm we choose.
             
    Output --
    """
    #choose the optimization algorithm, loss function and the track metric
    model.compile(optimizer = opt, loss = loss_fun, metrics = metrics)
    #fit the model with our training set and evaluate the model on test dat
    history = model.fit(xtrain, ytrain, batch_size = batch_size, 
                        epochs = n_epoch, callbacks = callback, 
                        validation_data = (xtest, ytest))
    
    return history
            
    
