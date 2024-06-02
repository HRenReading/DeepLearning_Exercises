# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:27:31 2024

@author: 44754
"""

import matplotlib.pyplot as plt
#import files
from utilis import *
from Convolution_TF import *

###########################################################################
"""
#plot settings
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=18)  # fontsize of the figure title
"""
###########################################################################
#load data and process it to the right data type and standardlization
#training set (x, y), test set(x, y), size of samples, widith and number
#of color channels, and number of classes
xtrain, ytrain, xtest, ytest, mtrain, size, K = process_data()

"Hyperparameters in Convolutional NN"
#filter height for 2 sets of filters
f = [4, 2]
#number of filters in each layers
n_f = [8, 16]
#stride size of convolution in each layer
s_conv = [1, 1]
#maxpool size
maxp = [8, 4]
#activation function for convolution
conv_fun = 'relu'
#optimizers used in CNN
opt1 = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, 
                               name="SGD")
opt2 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, 
                                beta_2=0.999, epsilon=1e-07, name="adam")
#mini-batch size
batch_size = 32
#early stop for training model
callback = [tf.keras.callbacks.EarlyStopping(monitor = "loss", 
                                             patience = 5,
                                             mode = "min",
                                             restore_best_weights = True)]
#loss function we choose to use
loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#number of epochs
n_epoch = 1000
#metrics tracer we use
metrics = ['accuracy']
########################################################################
"experiment of convolutional NN using tensorflow.keras"
#build the CNN
model = Model(n_f, f, size, maxp, s_conv, conv_fun, K)
history = CNN(model, opt2, xtrain, ytrain, xtest, ytest, loss_fun, 
              batch_size, n_epoch, callback, metrics)

plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label = 'Dev loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='training acc')
plt.plot(history.history['val_accuracy'], label = 'Dev acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()






