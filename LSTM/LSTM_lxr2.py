# -*- coding: utf-8 -*-
"""
Function: Train a LSTM model for fraud detection
Input: train.csv
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
from sklearn.utils import shuffle 
from sklearn.metrics import confusion_matrix


import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file
#os.chdir("/home/theano/src/small")

#lets not be annoyed by any warnings unless they are errors
import warnings
warnings.filterwarnings('ignore')

labelName="label"
cardname="card"
runEpoch=2

BS = 256
#runLoop = 50

Alldata = pd.read_csv('train.csv')
Alldata = shuffle(Alldata)

train_all,test_all=train_test_split(Alldata, test_size=0.2)


#train_all = pd.read_csv('train-converted_4.csv')
#test_all = pd.read_csv('train-converted_4.csv')



y_train = train_all.label
y_test = test_all.label

X_train = train_all.drop(labelName, axis = 1, inplace=False).drop(cardname, axis = 1, inplace=False)
X_test = test_all.drop(labelName, axis = 1, inplace=False).drop(cardname, axis = 1, inplace=False)
#X_train = X_train.drop(cardname, axis = 1, inplace=False)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

size_data = X_train.shape[1];
print "data dimension=", size_data
# Defining our classifier builder object to do all layers in once using layer codes from previous part



data_dim = size_data
timesteps = 4


def classifier_builder ():
    
    max_features = 256
    classifier = Sequential()
    classifier.add(Embedding(max_features, output_dim=256))
    #classifier.add(LSTM(128, input_shape=(timesteps, data_dim)))
    
    classifier.add(LSTM(128, input_shape=(data_dim, timesteps, )))
    
    classifier.add(Dropout(0.3))
    classifier.add(Dense(1, activation='sigmoid'))

    classifier.compile(loss='binary_crossentropy',
              optimizer='Adam', #'rmsprop',
              metrics=['accuracy'])

    return classifier


#Now we should create classifier object using our internal classifier object in the function above
classifier = KerasClassifier(build_fn= classifier_builder,
                             batch_size = 1024,
                             nb_epoch = runEpoch) #10)


classifier.fit(X_train, y_train, batch_size=1024, epochs=runEpoch)

y_predict=classifier.predict(X_test,batch_size=BS)
y_predict =  [j[0] for j in y_predict]
y_predict = np.where(np.array(y_predict)<0.5,0,1)

confusion_matrix=confusion_matrix(y_test,y_predict)
print  confusion_matrix

 