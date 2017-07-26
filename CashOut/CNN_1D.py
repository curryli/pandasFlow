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
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers.core import Dense, Dropout, Activation,Flatten

  
from keras.layers import Conv1D, MaxPooling1D, Embedding  
from keras.models import Model

 
from keras.layers import LSTM
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers import BatchNormalization
from sklearn.svm import SVC
 
from keras.utils import np_utils
from keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import recall_score, precision_score
from keras import metrics
import keras.backend as K
from sklearn.utils import shuffle 
from sklearn.metrics import confusion_matrix
from keras.preprocessing.sequence import pad_sequences  
import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file

#lets not be annoyed by any warnings unless they are errors
import warnings
warnings.filterwarnings('ignore')

labelName="label"
cardname="pri_acct_no_conv"
runEpoch=100

BS = 256
#runLoop = 50


EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 10



Alldata = pd.read_csv('idx_new_08_sort.csv')
Alldata = shuffle(Alldata)

train_all,test_all=train_test_split(Alldata, test_size=0.2)


#train_all = pd.read_csv('train-converted_4.csv')
#test_all = pd.read_csv('train-converted_4.csv')



y_train = train_all.label
y_test = test_all.label

X_train = train_all.drop(labelName, axis = 1, inplace=False).drop(cardname, axis = 1, inplace=False)
X_test = test_all.drop(labelName, axis = 1, inplace=False).drop(cardname, axis = 1, inplace=False)
#X_train = X_train.drop(cardname, axis = 1, inplace=False)
#X_test =  X_test.drop(cardname, axis = 1, inplace=False)


sc = StandardScaler()

#print X_train.loc[:1]

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)  
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)  
 


max_features = int(abs(np.amax(X_train)))+1 #X_train.shape[1];
print "Embedding max_features=", max_features 
  
# Defining our classifier builder object to do all layers in once using layer codes from previous part


size_data = X_train.shape[1];
print "data dimension=", size_data


embedding_layer = Embedding(max_features,
                            EMBEDDING_DIM, 
                            input_length=MAX_SEQUENCE_LENGTH)
  
# 建立序贯模型
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(128, 5, input_shape = (MAX_SEQUENCE_LENGTH,size_data)))
model.add(Activation('tanh'))  
model.add(MaxPooling1D(2))
model.add(Dropout(0.3))
#model.add(Conv1D(128, 5,input_shape = (MAX_SEQUENCE_LENGTH,size_data), activation='relu'))
#model.add(MaxPooling1D(2))
#model.add(Dropout(0.3))

# Flatten层，把多维输入进行一维化，常用在卷积层到全连接层的过渡
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

 
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          nb_epoch=2, batch_size=128)

 
y_predict=model.predict(X_test,batch_size=BS)
y_predict =  [j[0] for j in y_predict]
y_predict = np.where(np.array(y_predict)<0.5,0,1)
 
precision = precision_score(y_test, y_predict, average='macro') 
recall = recall_score(y_test,y_predict, average='macro') 
print ("Precision:", precision) 
print ("Recall:", recall) 

confusion_matrix=confusion_matrix(y_test,y_predict)
print  confusion_matrix

 