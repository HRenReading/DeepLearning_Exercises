# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:46:43 2024

@author: 44754
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import urllib


def download_data():
  train = 'https://github.com/csaybar/DLcoursera/raw/master/Neural%20Networks%20and%20Deep%20Learning/week2/Logistic%20Regression%20as%20a%20Neural%20Network/datasets/train_catvnoncat.h5'
  test = 'https://github.com/csaybar/DLcoursera/raw/master/Neural%20Networks%20and%20Deep%20Learning/week2/Logistic%20Regression%20as%20a%20Neural%20Network/datasets/test_catvnoncat.h5'
  urllib.request.urlretrieve(train,'train_catvnoncat.h5')
  urllib.request.urlretrieve(test,'test_catvnoncat.h5')

def load_data():
    
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def reshape(x):
    "reformulate the data to 2 dimensional, (n features,m examples)"
    #reshape the data to (n,m) (number of features,sample size)
    xnew = x.reshape(x.shape[0],-1).T
    #count the numbers of samples and features
    n,m = xnew.shape
    return xnew,m,n
