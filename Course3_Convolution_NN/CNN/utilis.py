# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:30:01 2024

@author: 44754
"""

import h5py
import numpy as np
import tensorflow as tf


######################################################################
"Data abstraction and processing"
def load_dataset():
    #read training data set
    train_dataset = h5py.File('train_signs.h5', "r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    # your train set labels
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    #read the test set data from file
    test_dataset = h5py.File('test_signs.h5', "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])#
    # your test set labels
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    #number of classes in the data sets
    classes = np.array(test_dataset["list_classes"][:])
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, \
           test_set_y_orig, classes


def process_data():
    """
    Process the image data, return the data (training and test) and 
    numbers of features and samples in the training set.
    """
     
    #load data from files
    xtrain,ytrain,xtest,ytest,classes = load_dataset()
    #count the numbers of examples in the training set
    mtrain = xtrain.shape[0]
    #find the size of a single image data
    size = xtrain.shape[1:]
    #convert feature data type to float and standardlize it
    xtrain = xtrain.astype(float)/255.
    xtest = xtest.astype(float)/255.
    #convert labels data type to int32
    ytrain = ytrain.astype(int)
    ytest = ytest.astype(int)
    #number of classes
    K = len(classes)
    #assign data to tensor
    Xtrain = tf.constant(xtrain, dtype = float)
    Xtest = tf.constant(xtest, dtype = float)
    #one-hot-encode our target variable
    #Ytrain = tf.one_hot(ytrain, depth = K)
    #Ytest = tf.one_hot(ytest, depth = K)
    
    return Xtrain,ytrain,Xtest,ytest,mtrain,size,K

#####################################################################

    
    
    
    