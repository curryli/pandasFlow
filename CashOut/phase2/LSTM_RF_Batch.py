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
 
from sklearn.ensemble import RandomForestClassifier 

import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#lets not be annoyed by any warnings unless they are errors
import warnings
warnings.filterwarnings('ignore')

labelName="label"
cardName = "pri_acct_no_conv" 
runEpoch=5

#modelName = "lstm_reshape_5.md"

BS = 128
#runLoop = 50

#Alldata = pd.read_csv('convert5_weika_GBDT_07.csv')
#Alldata = pd.read_csv('convert_5_card_GBDT.csv')
Alldata = pd.read_csv('convert_5_card.csv')




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
#Alldata = Alldata.head(10000)
 

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
 
    
    classifier.add(LSTM(128, input_shape=(timesteps, data_dim), recurrent_dropout=0.2, activation='sigmoid',  recurrent_activation='hard_sigmoid',   unit_forget_bias=True, return_sequences=True))
    classifier.add(Dropout(0.2))
    classifier.add(LSTM(64,  recurrent_dropout=0.2, activation='sigmoid',  recurrent_activation='hard_sigmoid',   unit_forget_bias=True))
    classifier.add(Dropout(0.2))
     
    classifier.add(Dense(32, activation='sigmoid'))
    classifier.add(Dropout(0.2))
    
    
    classifier.add(Dense(1, activation='sigmoid', kernel_constraint=maxnorm(2)))  #'tanh'  'sigmoid'
 
    
    
    
    classifier.compile(loss='binary_crossentropy',
              optimizer= 'Adam',                           #RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.01),                  #'rmsprop', #'rmsprop','Adam'
              metrics=[metrics.mae,"accuracy"])

    return classifier

class_weights= dict() 
class_weights[0] =  1     #这个是正常的
class_weights[1] = 20  #这个是欺诈的    

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
  
classifier.fit(X_train, y_train, batch_size=BS, epochs=runEpoch, class_weight='balanced',  validation_data=(X_test, y_test), verbose=2)


bk.set_learning_phase(0)  #测试阶段
print "After set ", bk.learning_phase()

y_predict=classifier.predict(X_test,batch_size=BS)
y_predict =  [j[0] for j in y_predict]
y_predict = np.where(np.array(y_predict)<0.5,0,1)
 
print  "\nconfusion_matrix:\n"
confusion_matrix_1=confusion_matrix(y_test,y_predict)
print  confusion_matrix_1


precision_p = float(confusion_matrix_1[1][1])/float((confusion_matrix_1[0][1] + confusion_matrix_1[1][1]))
recall_p = float(confusion_matrix_1[1][1])/float((confusion_matrix_1[1][0] + confusion_matrix_1[1][1]))
 
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 
F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)

print F1_Score 

if(os.access("lstm_model.h5", os.F_OK)):
    print(classifier.summary())
    f_dense1 = K.function([classifier.layers[0].input],   [classifier.layers[6].output])
else:
    print(classifier.model.summary())
    f_dense1 = K.function([classifier.model.layers[0].input],  [classifier.model.layers[6].output])



#############Transform train##############################
Read_batch = 5000
index_in_epoch = 0 
num_train = X_train.shape[0]
contents = X_train
labels = y_train.values.reshape(y_train.values.shape[0],1)
def get_next_batch(batch_size, index_in_epoch, contents, labels):
    start = index_in_epoch
    index_in_epoch = index_in_epoch + batch_size
     
    end = index_in_epoch
    return contents[start:end], labels[start:end]

#初始化#
(X_train,y_train) = get_next_batch(1, index_in_epoch, contents, labels)
layer_output_train = f_dense1([X_train])[0]
layer_output_train = np.hstack((layer_output_train,X_train[:, (timesteps-1)*data_dim: ]))
print  "layer_output_train.shape: ", layer_output_train.shape, "y_train.shape", y_train.shape
#初始化#

i=1
while(i<num_train):
        print i
        index_in_epoch = i
        (X_batch, y_batch) = get_next_batch(Read_batch, index_in_epoch, contents, labels)
        #print X_batch, y_batch
        
        i = i + Read_batch
        layer_output_batch = f_dense1([X_batch])[0]
        #把LSTM和随机森林特征结合
        layer_output_batch = np.hstack((layer_output_batch,X_batch[:, (timesteps-1)*data_dim: ]))
         
        layer_output_train = np.vstack((layer_output_train, layer_output_batch)) 
        y_train = np.vstack((y_train,y_batch)) 
       
 
print  "layer_output_train.shape: ", layer_output_train.shape, "y_train.shape", y_train.shape
#############Transform train############################## 

#############Transform test##############################

index_in_epoch = 0 
num_test = X_test.shape[0]
contents = X_test
labels = y_test.values.reshape(y_test.values.shape[0],1)
def get_next_batch(batch_size, index_in_epoch, contents, labels):
    start = index_in_epoch
    index_in_epoch = index_in_epoch + batch_size
     
    end = index_in_epoch
    return contents[start:end], labels[start:end]

#初始化#
(X_test,y_test) = get_next_batch(1, index_in_epoch, contents, labels)
layer_output_test = f_dense1([X_test])[0]
layer_output_test = np.hstack((layer_output_test,X_test[:, (timesteps-1)*data_dim: ]))
#print  "layer_output_test.shape: ", layer_output_test.shape, "y_test.shape", y_test.shape
#初始化#

i=1
while(i<num_test):
        print i
        index_in_epoch = i
        (X_batch, y_batch) = get_next_batch(Read_batch, index_in_epoch, contents, labels)
        #print X_batch, y_batch
        
        i = i + Read_batch
        layer_output_batch = f_dense1([X_batch])[0]
        #把LSTM和随机森林特征结合
        layer_output_batch = np.hstack((layer_output_batch,X_batch[:, (timesteps-1)*data_dim: ]))
         
        layer_output_test = np.vstack((layer_output_test, layer_output_batch)) 
        y_test = np.vstack((y_test,y_batch)) 
       
 
print  "layer_output_test.shape: ", layer_output_test.shape, "y_test.shape", y_test.shape
#############Transform test############################## 
 
X_train = layer_output_train
X_test = layer_output_test
  
#n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features="auto",max_leaf_nodes=None, bootstrap=True)

print  "start fit RF:\n"

clf = clf.fit(X_train, y_train)
  
print clf.score(X_test, y_test)

pred = clf.predict(X_test) 
 
#print type(y_test),type(pred)
#confusion_matrix=confusion_matrix(y_test.tolist(), pred.tolist())

#上面confusion_matrix要新起一个名字，否则会报错  TypeError: 'numpy.ndarray' object is not callable，和内部变量冲突了
confusion_matrix_2=confusion_matrix(y_test, pred)
print  "\nconfusion_matrix:\n"
print  confusion_matrix_2


precision_p = float(confusion_matrix_2[1][1])/float((confusion_matrix_2[0][1] + confusion_matrix_2[1][1]))
recall_p = float(confusion_matrix_2[1][1])/float((confusion_matrix_2[1][0] + confusion_matrix_2[1][1]))
 
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 
F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)

print F1_Score 
 
 



#_________________________________________________________________
#Layer (type)                 Output Shape                
#=================================================================
#reshape_1 (Reshape)          (None, 5, 67)             0        
#_________________________________________________________________
#masking_1 (Masking)          (None, 5, 67)             1         
#_________________________________________________________________
#lstm_1 (LSTM)                (None, 5, 128)            2    
#_________________________________________________________________
#dropout_1 (Dropout)          (None, 5, 128)            3         
#_________________________________________________________________
#lstm_2 (LSTM)                (None, 64)                4     
#_________________________________________________________________
#dropout_2 (Dropout)          (None, 64)                5        
#_________________________________________________________________
#dense_1 (Dense)              (None, 32)                6     
#_________________________________________________________________
#dropout_3 (Dropout)          (None, 32)                7         
#_________________________________________________________________
#dense_2 (Dense)              (None, 1)                 8       
#=================================================================