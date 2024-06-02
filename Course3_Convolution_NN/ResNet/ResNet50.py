# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:48:57 2024

@author: 44754
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, \
                                    Dense, ZeroPadding2D, Flatten,\
                                    Activation, BatchNormalization,\
                                    Add, AveragePooling2D

####################################################################
      
def CNN(size):
    """
    Standard CNN with Batch normalization
    
    Input -- size, size of a single image data.
    Output -- the input layer and standard CNN.
    """
    #Input layer
    x_inp = Input(size)
    #Padding layer for the original image data
    x = ZeroPadding2D((3,3))(x_inp)
    #Convolution layer
    x = Conv2D(64, kernel_size = 7, strides = (2, 2), 
               padding='valid', 
               kernel_initializer='glorot_uniform', 
               activation = None)(x)
    #Batch normalization layer
    x = BatchNormalization(axis = -1)(x)
    #activation function layer
    x = Activation(activation = 'relu')(x)
    #Maxpooling layer
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2),
                     padding = 'valid')(x)
    return x_inp, x
    
 
def Residual_block(x, f, s, filters):
    """
    Residual convolutional block with batch normalization.
    
    Input -- x, previous tensor block.
             f, height/width of the filter window.
             s, stride size.
             filters, a list of filters numbers.
    Output -- A Residual convolutional block.
    """    
    #save the previous output as a short cut
    shortcut = x
    "first mini-block of the main path"
    #convolution
    x = Conv2D(filters = filters[0], kernel_size = 1, 
               strides = (s, s), padding = 'valid', 
               kernel_initializer = 'glorot_uniform',
               activation = None)(x)
    #batch normalization
    x = BatchNormalization(axis = -1)(x)
    #Activation
    x = Activation('relu')(x)
    "second mini-block of the main path"
    #convolution
    x = Conv2D(filters = filters[1], kernel_size = f, 
               strides = (1, 1), padding = 'same', 
               kernel_initializer = 'glorot_uniform',
               activation = None)(x)
    #batch normalization
    x = BatchNormalization(axis = -1)(x)
    #Activation
    x = Activation('relu')(x)
    "third mini-block of the main path, only convolution"
    "and normalization"
    #convolution
    x = Conv2D(filters = filters[2], kernel_size = 1, 
               strides = (1, 1), padding = 'valid',
               kernel_initializer = 'glorot_uniform',
               activation = None)(x)
    #batch normalization
    x = BatchNormalization(axis = -1)(x)
    "short cut path, convolution and normalization"
    #convolution
    shortcut = Conv2D(filters = filters[2], kernel_size = 1,
                      strides = (s, s), padding = 'valid',
                      kernel_initializer = 'glorot_uniform',
                      activation = None)(shortcut)
    #batch normalization
    shortcut = BatchNormalization(axis = -1)(shortcut)
    #add the shortcut to the main path
    x = Add()([x,shortcut])
    #activation for the sum
    x = Activation('relu')(x)

    return x
    
 
def indentity_block(x, f, filters):
    """
    Indentity block (Residual conv block when input and 
                     output are the same size/shape.)
    
    Input -- 
    """
    #save the previous output as a short cut
    shortcut = x
    "first mini-block of the main path"
    #convolution
    x = Conv2D(filters = filters[0], kernel_size = 1, 
               strides = (1, 1), padding = 'valid', 
               kernel_initializer = 'glorot_uniform',
               activation = None)(x)
    #batch normalization
    x = BatchNormalization(axis = -1)(x)
    #Activation
    x = Activation('relu')(x)
    "second mini-block of the main path"
    #convolution
    x = Conv2D(filters = filters[1], kernel_size = f, 
               strides = (1, 1), padding = 'same', 
               kernel_initializer = 'glorot_uniform',
               activation = None)(x)
    #batch normalization
    x = BatchNormalization(axis = -1)(x)
    #Activation
    x = Activation('relu')(x)
    "third mini-block of the main path, only convolution"
    "and normalization"
    #convolution
    x = Conv2D(filters = filters[2], kernel_size = 1, 
               strides = (1, 1), padding = 'valid',
               kernel_initializer = 'glorot_uniform',
               activation = None)(x)
    #batch normalization
    x = BatchNormalization(axis = -1)(x)
    #add the short cut
    x = Add()([x,shortcut])
    #activation for the sum
    x = Activation('relu')(x)
    
    return x
    
####################################################################

def ResNet50(size, classes):
    """
    50-layer deepl ResNet.
    
    Input -- xtrain, training data set features.
             
    Output -- 
    """
    "Stage 1. standard CNN with batch normalization"
    #compute the initial input and the standard CNN with 
    #batch normallization
    x_inp, x = CNN(size)
    "Stage 2. 1 Residual block and 2 identity blocks"
    #residual convolutional block
    x = Residual_block(x, 3, 1, [64, 64, 256])
    #2 identity blocks (input and output are the same size)
    for i in range(2):
        x = indentity_block(x, 3, [64, 64, 256])
        del i
    "Stage 3. 1 Residual block and 3 identity blocks and a"
    "average pooling layer"
    #residual convolutional block
    x = Residual_block(x, 3, 2, [128, 128, 512])
    #3 indentity blocks
    for i in range(3):
        x = indentity_block(x, 3, [128, 128, 512])
        del i
    "Stage 4. 1 Residual block and 5 identity blocks"
    #residual convolutional block
    x = Residual_block(x, 3, 2, [256, 256, 1024])
    #5 indentity blocks
    for i in range(5):
        x = indentity_block(x, 3, [256, 256, 1024])
        del i
    "Stage 5. 1 Residual block and 2 identity blocks, and "
    "1 averagepooling layer"
    #residual convolutional block
    x = Residual_block(x, 3, 2, [512, 512, 2048])
    #2 identity blocks
    for i in range(2):
        x = indentity_block(x, 3, [512, 512, 2048])
        del i
    #average pooling layer
    x = AveragePooling2D(pool_size = (2, 2), strides = (1, 1))(x)
    "Final state. Fully connected layer"
    #flatten layer
    x = Flatten()(x)
    #fully connected layer
    x = Dense(classes, activation = 'softmax', 
              kernel_initializer = 'glorot_uniform')(x)
    
    #build the structure of the ResNet
    model = Model(inputs = x_inp, outputs = x)
    
    return model
    
    
    

