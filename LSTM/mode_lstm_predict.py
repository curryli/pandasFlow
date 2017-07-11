# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:27:34 2017

@author: yuwei
"""

import pandas as pd                #data analysing tool for python
import matplotlib.pyplot as plt    #data visualising tool for python
import numpy as np
np.random.seed(1234)

from IPython.display import display
from sklearn.preprocessing import StandardScaler                   # for normalization of our data
from keras.wrappers.scikit_learn import KerasClassifier            #package allowing keras to work with python
from sklearn.model_selection import cross_val_score, GridSearchCV  #using Kfold and if needed, GridSearch object in analysis
from sklearn.utils import shuffle                                  # shuffling our own made dataset
from keras.models import Sequential                                # linear layer stacks model for keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers import BatchNormalization
from sklearn.svm import SVC
import theano
from keras.utils import np_utils
from keras.models import load_model

import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file
#os.chdir("/home/theano/src/small")


labelName="label"
cardname="card"
batchsize=1 #16
train_all = pd.read_csv('train.csv')
test_all = pd.read_csv('test.csv')

y_train = train_all.label
y_test = test_all.label


X_train = train_all.drop(labelName, axis = 1)
X_test = test_all.drop(labelName, axis = 1)
X_train = train_all.drop(cardname, axis = 1)
X_test = test_all.drop(cardname, axis = 1)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier=load_model('lstm_model.h5')
loss_and_metrics=classifier.evaluate(X_test, y_test, batch_size=batchsize,verbose=1)
print loss_and_metrics
loss_and_metrics2=classifier.evaluate(X_train, y_train, batch_size=batchsize,verbose=1)
print loss_and_metrics2


y_predict_train=classifier.predict(X_train, batch_size=1, verbose=1)
y_predict_test =classifier.predict(X_test, batch_size=1, verbose=1)