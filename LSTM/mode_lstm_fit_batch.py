# -*- coding: utf-8 -*-
"""
Function: Train a LSTM model for fraud detection as batch by batch
Input: train_fraud.csv/train_clean.csv after sc
Output: lstm_model.h5 model file

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

#lets not be annoyed by any warnings unless they are errors
import warnings
warnings.filterwarnings('ignore')

labelName="label"
cardname="card"
batchSize = 2
runEpoch=5

X_train_fraud= pd.read_csv('train_fraud.csv')
X_train_clean= pd.read_csv('train_clean.csv')

y_train_fraud = pd.Series(np.ones(X_train_fraud.shape[0]))
y_train_clean = pd.Series(np.zeros(X_train_clean.shape[0]))

X_train_fraud = X_train_fraud.as_matrix()
X_train_clean = X_train_clean.as_matrix()

def classifier_builder ():
    
    max_features = 256 #size_data #100
    classifier = Sequential()
    classifier.add(Embedding(max_features, output_dim=256))
    classifier.add(LSTM(128))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(1, activation='sigmoid'))

    classifier.compile(loss='binary_crossentropy',
              optimizer='Adam', #'rmsprop',
              metrics=['accuracy'])

    return classifier




#Now we should create classifier object using our internal classifier object in the function above
classifier = KerasClassifier(build_fn= classifier_builder,
                             batch_size = 1024, #16,
                             nb_epoch = 1) #10)

sizeX = X_train_clean.shape[0]
sizeKf= X_train_fraud.shape[0]
maxBatch= int(sizeX / batchSize)
print "maxBatch=", maxBatch, "sizeX=", sizeX
X_train_batch = X_train_clean[1:(2*batchSize+1)]
y_train_batch = y_train_clean[1:(2*batchSize+1)]

isLoad=os.access("lstm_model.h5", os.F_OK)
if(isLoad):
    classifier=load_model('lstm_model.h5')
    print "load model"
else:
    hist=classifier.fit(X_train_batch, y_train_batch, batch_size=1, epochs=1)
    print "fit model"
    print(hist.history)


maxEpoch= 3
for e in range(0, maxEpoch):
	print "current epoch=", e, "of", maxEpoch
	k=0
	kf=0
	for b in range(0, maxBatch):
		for i in range(0, batchSize):
			X_train_batch[i] = X_train_clean[k]
			X_train_batch[i+batchSize] = X_train_fraud[kf]
			y_train_batch[i+1] = y_train_clean[k]
			y_train_batch[i+1+batchSize] = y_train_fraud[kf]
			k=k+1
			kf=kf+1
			if(kf >= sizeKf):
				kf = 0
			if(isLoad):
			    classifier.train_on_batch(X_train_batch, y_train_batch)
			else:
			    classifier.model.train_on_batch(X_train_batch, y_train_batch)
		print b, " of ", maxBatch
	

if(isLoad):
    print(classifier.summary()) 
    classifier.save('lstm_model.h5')
else:
    print(classifier.model.summary())
    classifier.model.save('lstm_model.h5')
