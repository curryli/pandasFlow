# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 08:30:44 2017

@author: lixurui
"""

from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model
import numpy as np
from keras import regularizers

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
 
#导入随机森林算法库
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle 
 
from sklearn.preprocessing import StandardScaler,MinMaxScaler                   # for normalization of our data
from keras.wrappers.scikit_learn import KerasClassifier            #package allowing keras to work with python
from sklearn.model_selection import cross_val_score, GridSearchCV  #using Kfold and if needed, GridSearch object in analysis
from sklearn.utils import shuffle                                  # shuffling our own made dataset
from keras.models import Sequential                                # linear layer stacks model for keras
 
from keras.layers import Dense, Dropout
from keras.layers.core import  Flatten, Reshape
from keras.layers import Embedding, Masking
 
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers import BatchNormalization
from sklearn.svm import SVC
 
from keras.utils import np_utils
from keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import recall_score 
from keras import metrics
import keras.backend as K
from sklearn.utils import shuffle 
 
from keras.layers.wrappers import Bidirectional
 
from sklearn.ensemble import RandomForestClassifier   
 
from keras.models import Sequential
import os  

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

labelName="label"
cardName = "pri_acct_no_conv" 
runEpoch=50
AEepochs = 10
#modelName = "lstm_reshape_5.md"

BS = 128
#runLoop = 50

#Alldata = pd.read_csv('convert_5_card_GBDT.csv')
Alldata = pd.read_csv('convert_5_card.csv')
#Alldata = pd.read_csv('convert_5_card_more.csv')




Fraud = Alldata[Alldata.label == 1]
Normal = Alldata[Alldata.label == 0]
print "Ori Fraud shape:", Fraud.shape, "Ori Normal shape:", Normal.shape

card_c_F = Fraud['pri_acct_no_conv'].drop_duplicates()#涉欺诈交易卡号

#未出现在欺诈样本中的正样本数据
fine_N=Normal[(~Normal['pri_acct_no_conv'].isin(card_c_F))]
print "True Fraud shape:", Fraud.shape, "True Normal shape:", fine_N.shape

Alldata = pd.concat([Fraud,fine_N], axis = 0)


###############################
Alldata = shuffle(Alldata)
print "Total samples:", Alldata.shape[0]
train_all,test_all=train_test_split(Alldata, test_size=0.2)


#train_all = pd.read_csv('train-converted_4.csv')
#test_all = pd.read_csv('train-converted_4.csv')



y_train = train_all.label
y_test = test_all.label

X_Train = train_all.drop(labelName, axis = 1, inplace=False).drop(cardName, axis = 1, inplace=False) 
X_Test = test_all.drop(labelName, axis = 1, inplace=False).drop(cardName, axis = 1, inplace=False)  



size_data = X_Train.shape[1];
print "size_data= ", size_data

timesteps = 5
data_dim = size_data/timesteps
print "data dimension=", data_dim
sc = StandardScaler()

#print X_train.loc[:1]

X_train = sc.fit_transform(X_Train)
X_test = sc.transform(X_Test)

X_test_v = X_test.reshape(X_Test.shape[0], timesteps, data_dim)
print X_test_v.shape

 
#autoencoder为：
Autoencoder = Sequential()
Autoencoder.add(Reshape((timesteps, data_dim), input_shape=(size_data,)))
Autoencoder.add(Masking(mask_value= -1, input_shape=(timesteps, data_dim)))
    
    
Autoencoder.add(LSTM(128, input_shape=(timesteps, data_dim)))
Autoencoder.add(RepeatVector(timesteps))
 
Autoencoder.add(LSTM(size_data, return_sequences=False))  
 
 
#Autoencoder.compile(optimizer='rmsprop', loss='mse', metrics=[metrics.mae,"accuracy"])
Autoencoder.compile(optimizer='rmsprop', loss='mse', metrics=["accuracy"])
#Autoencoder.compile(optimizer='adam', loss='mse')
#进行训练：
Autoencoder.fit(X_train, X_train, nb_epoch=AEepochs, shuffle=True, validation_data=(X_test, X_test))
 
  