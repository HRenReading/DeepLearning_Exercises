# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:27:01 2024

@author: 44754
"""
import numpy as np
import h5py


def readHDF(filename):
    "read data from a HDF file and put it into ndarray"
    #Open the H5 file in read mode
    with h5py.File(filename, 'r') as file:
        #set the key of features of all examples
        x_key = list(file.keys())[1]
        #set the key of labels
        y_key = list(file.keys())[2]
        # Getting the data
        x_train = list(file[x_key])
        y_train = list(file[y_key])
        #count the size of the data
        m = len(y_train)
        #put labels data into ndarray
        y = np.array(y_train)
        #find the size of single examples
        x0 = x_train[0]
        size = x0.shape
        #put data into a ndarray
        x = np.zeros((m,size[0]*size[1]*size[2]))
        for i in range(m):
            x[i,:] = x_train[i].astype(float).ravel()
            del i
        n = x.shape[1]
            
        return x,y,m,n,size
    
    
def convert_uint8(x,size,m):
    "convert the data type from int or float to uint8, and the original shape \
    for plotting (pix,pix,colors)"
    #set empty matrix for storing the uint8 data
    img = np.zeros((m,size[0],size[1],size[2]))
    for i in range(m):
        img[i,:,:,:] = x[i,:].reshape(size[0],size[1],size[2])
        del i
    return np.uint8(img)
    
    
    
    