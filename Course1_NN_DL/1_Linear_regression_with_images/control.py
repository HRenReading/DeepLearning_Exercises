# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:30:49 2024

@author: 44754
"""

#import files
from lr_utilis import *
from Logistic_regression import *

#learn from the training data
x,y,theta,error = Logistic_regression('train_catvnoncat.h5')
#read the test data
xtest,ytest,mtest,ntest,testsize = readHDF('test_catvnoncat.h5')
#standardlize the test data
xtest /= 255.
test_pred = prediction(xtest,mtest,ntest,theta)
errortest = np.sum((test_pred - ytest)**2)/mtest
print('The accuracy of our trained model is',(1-errortest)*100,'%')
