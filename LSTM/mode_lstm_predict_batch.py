# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:27:34 2017

@author: yuwei
"""

import pandas as pd                #data analysing tool for python
import matplotlib.pyplot as plt    #data visualising tool for python
import numpy as np
from IPython.display import display
from sklearn.preprocessing import StandardScaler                   # for normalization of our data
from keras.wrappers.scikit_learn import KerasClassifier            #package allowing keras to work with python
from sklearn.model_selection import cross_val_score, GridSearchCV  #using Kfold and if needed, GridSearch object in analysis
from sklearn.utils import shuffle                                  # shuffling our own made dataset
from keras.models import Sequential                                # linear layer stacks model for keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout

import os                          #python miscellaneous OS system tool

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

#os.chdir("/home/theano/src/small")

#lets not be annoyed by any warnings unless they are errors
import warnings
warnings.filterwarnings('ignore')

##############################
# Loading data
##############################
isTestFraud = True
isTestClean = True
isTestAll = False

labelName="label"
cardname = "card"



train_all = pd.read_csv('train.csv')
test_all = pd.read_csv('test.csv')
y_train = train_all.label #"label"
y_test  = test_all.label  #"label"
X_train = train_all.drop(labelName, axis = 1)
X_test  = test_all.drop(labelName, axis = 1)
X_train = train_all.drop(cardname, axis = 1)
X_test  = test_all.drop(cardname, axis = 1)

X_test_fraud= pd.read_csv('test_fraud.csv')
X_test_clean= pd.read_csv('test_clean.csv')
#X_test_fraud= test_fraud.drop(labelName, axis = 1)
#X_test_clean= test_clean.drop(labelName, axis = 1)
y_test_fraud=  pd.Series(np.ones(X_test_fraud.shape[0]))
y_test_clean=  pd.Series(np.zeros(X_test_clean.shape[0]))

X_test_fraud = X_test_fraud.as_matrix()
X_test_clean = X_test_clean.as_matrix()
###############################
# Normalization
###############################
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



###############################
# For debug only
###############################
def comp(yp, yt):
	leny = len(yp)
	print leny
	k=0
	for i in range(0,leny):
	    if abs(float(yp[i]) - float(yt[i])) >= 0.5:
	        print i, yp[i], yt[i]
		k=k+1
	print 'diff number=', k
	return leny

# Defining our classifier builder object to do all layers in once using layer codes from previous part

classifier=load_model('lstm_model.h5')

if isTestAll:
	loss_and_metrics=classifier.evaluate(X_test, y_test, batch_size=1,verbose=1)
	print loss_and_metrics
	loss_and_metrics2=classifier.evaluate(X_train, y_train, batch_size=1,verbose=1)
	print loss_and_metrics2

if isTestFraud:
	print "start fraud test ..."
	loss_and_metrics=classifier.evaluate(X_test_fraud, y_test_fraud, batch_size=1,verbose=1)
	print "fraud =", loss_and_metrics

if isTestClean:
	print "start clean test ..."
	loss_and_metrics=classifier.evaluate(X_test_clean, y_test_clean, batch_size=1,verbose=1)
	print "clean =", loss_and_metrics

