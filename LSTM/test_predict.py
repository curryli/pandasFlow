# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:27:34 2017

@author: yuwei
"""

import pandas as pd                #data analysing tool for python
import matplotlib.pyplot as plt    #data visualising tool for python
import numpy as np
np.random.seed(1234)

from keras import metrics
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
from sklearn.metrics import recall_score, precision_score

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
 
#导入随机森林算法库
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
 

import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file
#os.chdir("/home/theano/src/small")


labelName="label"
cardname="card"
batchsize=256 #16
train_all = pd.read_csv('train.csv')
test_all = pd.read_csv('test2.csv')
 
y_test = np.array(test_all.label.map(lambda x: int(x)), dtype=np.int32)


X_train = train_all.drop(labelName, axis = 1, inplace=False)
X_train = X_train.drop(cardname, axis = 1, inplace=False)
X_train = pd.get_dummies(X_train)


X_test = test_all.drop(labelName, axis = 1, inplace=False)
X_test = X_test.drop(cardname, axis = 1, inplace=False)
X_test = pd.get_dummies(X_test)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier=load_model('lstm_model.h5')

predict=classifier.predict(X_test,batch_size=batchsize) #输出预测结果
predict =  [i[0] for i in predict]
print y_test[:10]
 
predict = np.where(np.array(predict)<0.5,0,1)
print predict[:10]

confusion_matrix=confusion_matrix(y_test,predict)
print  confusion_matrix

precision = precision_score(y_test, predict, average='macro') 
recall = recall_score(y_test, predict, average='macro') 
print ("Precision:", precision) 
print ("Recall:", recall) 