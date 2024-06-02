# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:02:19 2024

@author: 44754
"""

import matplotlib.pyplot as plt
from utilis import *
from ResNet50 import *

#########################################################################

#load images data from file, standardlize them, put the data into tensors
#find the number of examples in training set, the size of a image 
#(pix, pix, color_channels), and the number of classes
xtrain, ytrain, xtest, ytest, mtrain, size, K = process_data()


"Hyperparameters"
#optimizers used in CNN
opt1 = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, 
                               name="SGD")
opt2 = tf.keras.optimizers.Adam(learning_rate=0.00015, beta_1=0.9, 
                                beta_2=0.999, epsilon=1e-07, name="adam")
#mini-batch size
batch_size = 32

#number of epochs
n_epoch = 20
#metrics tracer we use
metrics = ['accuracy']

##########################################################################
"experiment with ResNet50"
#build the ResNet model
model = ResNet50(size, K)

#set the optimizer and loss function and the metrics we want
model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = metrics)


#train the model with training set
history = model.fit(xtrain, tf.one_hot(ytrain, K), 
                    batch_size = batch_size, 
                    epochs = n_epoch, 
                    validation_data = (xtest, tf.one_hot(ytest, K)),
                    validation_batch_size = batch_size,
                    validation_freq = 1)


##########################################################################
#plot the loss and acc of the training set and test set
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


