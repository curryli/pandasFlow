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
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import recall_score, precision_score
from keras import metrics
import keras.backend as K
from sklearn.utils import shuffle 

import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file

#lets not be annoyed by any warnings unless they are errors
import warnings
warnings.filterwarnings('ignore')

labelName="label"
cardname="card"
runEpoch=50

BS = 8
#runLoop = 50

Alldata = pd.read_csv('train-converted_4.csv')
Alldata = shuffle(Alldata)

train_all,test_all=train_test_split(Alldata, test_size=0.2)


#train_all = pd.read_csv('train-converted_4.csv')
#test_all = pd.read_csv('train-converted_4.csv')



y_train = train_all.label
y_test = test_all.label

X_train = train_all.drop(labelName, axis = 1, inplace=False)
X_test = test_all.drop(labelName, axis = 1, inplace=False)
#X_train = X_train.drop(cardname, axis = 1, inplace=False)
#X_test =  X_test.drop(cardname, axis = 1, inplace=False)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

global size_data
size_data = int(abs(np.amax(X_train)))+1 #X_train.shape[1];
print "Embedding max_features=", size_data
# Defining our classifier builder object to do all layers in once using layer codes from previous part

def classifier_builder ():
    
    max_features = size_data #256
    classifier = Sequential()
    classifier.add(Embedding(max_features, output_dim=256))
    classifier.add(LSTM(128))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(1, activation='sigmoid'))

    classifier.compile(loss='binary_crossentropy',
              optimizer='Adam', #'rmsprop',
              metrics=[metrics.mae,"accuracy"])

    return classifier

class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
print "class weights = ", class_weights

#Now we should create classifier object using our internal classifier object in the function above
classifier = KerasClassifier(build_fn= classifier_builder,
                             batch_size = BS,
                             nb_epoch = 1) 

 
 
classifier.fit(X_train, y_train, batch_size=BS, epochs=runEpoch, class_weight=class_weights, validation_data=(X_test, y_test))

 

y_predict=classifier.predict(X_test,batch_size=BS)
y_predict =  [j[0] for j in y_predict]
y_predict = np.where(np.array(y_predict)<0.5,0,1)
 
precision = precision_score(y_test, y_predict, average='macro') 
recall = recall_score(y_test,y_predict, average='macro') 
print ("Precision:", precision) 
print ("Recall:", recall) 

 
print(classifier.model.summary())
classifier.model.save('lstm_model.h5')

