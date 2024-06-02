# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:26:11 2024

@author: 44754
"""

import h5py
import numpy as np
import tensorflow as tf

def load_dataset():
    train_dataset = h5py.File('train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig.T, test_set_x_orig, test_set_y_orig.T, classes


def convert(x):
    """
    Covert the data from high-dimensional to 2 dimensional (m,n)
    
    Input -- x, data features.
    Output -- xnew, the reshaped data (2D).
    """
    #find the number of examples in the data set
    m = int(x.shape[0])
    #calculate the number of features 
    n = x.shape[1] * x.shape[2] * x.shape[3]
    #reshape the data from high-dimensional to 2D (m,n)
    xnew = x.reshape(m,n)
    
    return xnew,m,n
    
    