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
from sklearn.preprocessing import StandardScaler,MinMaxScaler                   # for normalization of our data
from keras.wrappers.scikit_learn import KerasClassifier            #package allowing keras to work with python
from sklearn.model_selection import cross_val_score, GridSearchCV  #using Kfold and if needed, GridSearch object in analysis
from sklearn.utils import shuffle                                  # shuffling our own made dataset
from keras.models import Sequential                                # linear layer stacks model for keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.layers.core import Dense, Dropout, Activation,Flatten, Reshape
from keras.layers import Embedding, Masking
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
from sklearn.metrics import confusion_matrix
from keras import regularizers
from keras.layers.wrappers import Bidirectional

import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file

#lets not be annoyed by any warnings unless they are errors
import warnings
warnings.filterwarnings('ignore')

labelName="label" 
runEpoch=200

modelName = "lstm_reshape_5.md"

BS = 128
#runLoop = 50

Alldata = pd.read_csv('cashout_3.csv')
Alldata = shuffle(Alldata)

print "Total samples:", Alldata.shape[0]
train_all,test_all=train_test_split(Alldata, test_size=0.2)


#train_all = pd.read_csv('train-converted_4.csv')
#test_all = pd.read_csv('train-converted_4.csv')



y_train = train_all.label
y_test = test_all.label

X_train = train_all.drop(labelName, axis = 1, inplace=False) 
X_test = test_all.drop(labelName, axis = 1, inplace=False) 

print X_train.columns

print len(X_train.columns)

size_data = X_train.shape[1];
print "size_data= ", size_data

timesteps = 3
data_dim = size_data/timesteps
print "data dimension=", data_dim


sc = StandardScaler()    #MinMaxScaler()    不好

#print X_train.loc[:1]

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

 
# Defining our classifier builder object to do all layers in once using layer codes from previous part

from keras.constraints import maxnorm
 
def classifier_builder ():
 
    classifier = Sequential()
    classifier.add(Reshape((timesteps, data_dim), input_shape=(size_data,)))
    classifier.add(Masking(mask_value= -1, input_shape=(timesteps, data_dim)))
    #classifier.add(LSTM(128))#               , input_shape=(timesteps, data_dim)))
    
#    classifier.add(Bidirectional(LSTM(32, input_shape=(timesteps, data_dim), recurrent_dropout=0.5, activation='sigmoid',  recurrent_activation='hard_sigmoid',   unit_forget_bias=True, return_sequences=True)))
#    classifier.add(Dropout(0.8))
#    classifier.add(Bidirectional(LSTM(32, input_shape=(timesteps, data_dim), recurrent_dropout=0.5, activation='sigmoid',  recurrent_activation='hard_sigmoid',   unit_forget_bias=True)))
#    classifier.add(Dropout(0.8))
    
    #classifier.add(LSTM(128, input_shape=(timesteps, data_dim),  dropout_W=0.7, dropout_U=0.7,   activation='relu',  recurrent_activation='hard_sigmoid',   unit_forget_bias=True, return_sequences=True))
    #classifier.add(Dropout(0.7))
    #classifier.add(LSTM(128, input_shape=(timesteps, data_dim),  dropout_W=0.7, dropout_U=0.7,   activation='relu',  recurrent_activation='hard_sigmoid',   unit_forget_bias=True))   # activation='sigmoid', recurrent_activation='hard_sigmoid', unit_forget_bias=True, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01), activity_regularizer=regularizers.l1(0.01)))
    #classifier.add(Dropout(0.7))
    
    
    
    
    #classifier.add(LSTM(128, input_shape=(timesteps, data_dim),  dropout=0.7, recurrent_dropout=0.7,  activation='sigmoid',  recurrent_activation='hard_sigmoid',   unit_forget_bias=True, return_sequences=True))
    #classifier.add(LSTM(128, input_shape=(timesteps, data_dim),  dropout=0.7, recurrent_dropout=0.7, activation='sigmoid',  recurrent_activation='hard_sigmoid',   unit_forget_bias=True))   # activation='sigmoid', recurrent_activation='hard_sigmoid', unit_forget_bias=True, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01), activity_regularizer=regularizers.l1(0.01)))
    
    
    
    
    classifier.add(LSTM(128, input_shape=(timesteps, data_dim), recurrent_dropout=0.3, activation='sigmoid',  recurrent_activation='hard_sigmoid',   unit_forget_bias=True, return_sequences=True))
    classifier.add(Dropout(0.7))
    classifier.add(LSTM(64,  recurrent_dropout=0.3, activation='sigmoid',  recurrent_activation='hard_sigmoid',   unit_forget_bias=True))
    classifier.add(Dropout(0.7))
     
    classifier.add(Dense(32, activation='sigmoid'))
    classifier.add(Dropout(0.7))
    
    
    classifier.add(Dense(1, activation='sigmoid', kernel_constraint=maxnorm(2)))  #'tanh'  'sigmoid'
 
    
    
    
    classifier.compile(loss='binary_crossentropy',
              optimizer= 'Adam',                           #RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.01),                  #'rmsprop', #'rmsprop','Adam'
              metrics=[metrics.mae,"accuracy"])

    return classifier

class_weights= dict() 
class_weights[0] =  1     #这个是正常的
class_weights[1] = 10  #这个是欺诈的    

#class_weights_2 = compute_class_weight('balanced', np.unique(y_train), y_train)
#class_weights= dict()
#for key in {0,1}:
#        class_weights[key] = class_weights_2[key]


#Now we should create classifier object using our internal classifier object in the function above
classifier = KerasClassifier(build_fn= classifier_builder,
                             batch_size = BS,
                             nb_epoch = runEpoch) 
 

#if(os.access(modelName, os.F_OK)):
#    classifier=load_model(modelName)

from keras import backend as bk
print "Before set", bk.learning_phase()


bk.set_learning_phase(1)   #训练阶段
print "After set ", bk.learning_phase()

#classifier.fit(X_train, y_train, batch_size=BS, epochs=runEpoch, class_weight=class_weights,  validation_data=(X_test, y_test), verbose=2)
  
classifier.fit(X_train, y_train, batch_size=BS, epochs=runEpoch,  validation_data=(X_test, y_test), verbose=2)


bk.set_learning_phase(0)  #测试阶段
print "After set ", bk.learning_phase()

y_predict=classifier.predict(X_test,batch_size=BS)
y_predict =  [j[0] for j in y_predict]
y_predict = np.where(np.array(y_predict)<0.5,0,1)
 
confusion_matrix=confusion_matrix(y_test,y_predict)
print  confusion_matrix


precision_p = float(confusion_matrix[1][1])/float((confusion_matrix[0][1] + confusion_matrix[1][1]))
recall_p = float(confusion_matrix[1][1])/float((confusion_matrix[1][0] + confusion_matrix[1][1]))
 
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 

if(os.access("lstm_lxr.md", os.F_OK)):
    print(classifier.summary()) 
    classifier.save('lstm_lxr.md')
else:
    print(classifier.model.summary())
    classifier.model.save('lstm_lxr.md')