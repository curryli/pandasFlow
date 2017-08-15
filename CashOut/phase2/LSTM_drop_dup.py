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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file
  
#lets not be annoyed by any warnings unless they are errors
import warnings
warnings.filterwarnings('ignore')

labelName="label"
cardName = "pri_acct_no_conv" 
runEpoch=200

#modelName = "lstm_reshape_5.md"

BS = 128
#runLoop = 50

#Alldata = pd.read_csv('convert_5_card.csv')
Alldata = pd.read_csv('convert_5_card_more.csv')




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



df_T,df_V=train_test_split(Alldata, test_size=0.2)
print  "df_T size: ", df_T.shape[0],"df_V size: ",df_V.shape[0]

#采样
fraud_size = 1
normal_size = 1

Fraud_T = df_T[Alldata.label == 1]
Fraud_train=Fraud_T
for i in range(0,fraud_size-1):
    Fraud_train=pd.concat([Fraud_train,Fraud_T], axis = 0)

normal_T = df_T[Alldata.label == 0]
normal_train=normal_T
for i in range(0,normal_size-1):
    normal_train=pd.concat([normal_train,normal_T], axis = 0)

print "Fraud_train_o size", Fraud_T.shape[0], "normal_train_o size",normal_T.shape[0]
print "Fraud_train size", Fraud_train.shape[0], "normal_train size",normal_train.shape[0]

df_train = pd.concat([Fraud_train,normal_train], axis = 0)
 
 

x_columns = [x for x in df_train.columns if x not in [labelName,cardName]]
X_Train = df_train[x_columns]
y_train = df_train[labelName]

#print "datasets size:%d" % X_train.shape[0]


X_Test = df_V[x_columns]
y_test = df_V[labelName]



size_data = X_Train.shape[1];
print "size_data= ", size_data

timesteps = 5
data_dim = size_data/timesteps
print "data dimension=", data_dim


sc = StandardScaler()    #MinMaxScaler()    不好

#print X_train.loc[:1]

X_train = sc.fit_transform(X_Train)
X_test = sc.transform(X_Test)

 
# Defining our classifier builder object to do all layers in once using layer codes from previous part

from keras.constraints import maxnorm
 
def classifier_builder ():
 
    classifier = Sequential()
    classifier.add(Reshape((timesteps, data_dim), input_shape=(size_data,)))
    classifier.add(Masking(mask_value= -1, input_shape=(timesteps, data_dim)))
 
    
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
class_weights[0] =  200     #这个是正常的
class_weights[1] = 1  #这个是欺诈的    

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

classifier.fit(X_train, y_train, batch_size=BS, epochs=runEpoch, class_weight=class_weights,  validation_data=(X_test, y_test), verbose=2)
  
#classifier.fit(X_train, y_train, batch_size=BS, epochs=runEpoch,  validation_data=(X_test, y_test), verbose=2)


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
    
    
#('Precision:', 0.9660511363636364)
#('Recall:', 0.9601863617111394)
 