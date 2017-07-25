# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:41:16 2017

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
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers import BatchNormalization
from sklearn.svm import SVC
from keras.utils import np_utils
from keras.models import load_model
import time

labelName="label"
cardname="card"
global transcationsPerRow
transcationsPerRow=2

start = time.clock()

allsamples = pd.read_csv('train.csv')
sizeSamples= allsamples.shape[0]

## sort allsamples by card/time if needed
print "Sorting according to card and time ..."
allsamples = allsamples.sort_values(by=[cardname, 'sus_1'])#sort_index
## end sort

global dropcardlabel
dropcardlabel=allsamples.drop(cardname, axis = 1, inplace=False)
dropcardlabel=dropcardlabel.drop(labelName, axis = 1, inplace=False)

global sizeX
sizeX= dropcardlabel.shape[1]
cardcol=pd.DataFrame(allsamples, columns=[cardname])
labelcol=pd.DataFrame(allsamples, columns=[labelName])

global mergedData
mergedData=pd.DataFrame(dropcardlabel)
dropcardlabelX = pd.DataFrame(dropcardlabel)
oneTransColName = pd.Index(dropcardlabel.columns)
for i in range (1, transcationsPerRow):
    mergedData= pd.concat([mergedData, dropcardlabelX], axis=1)
 
global zeropad
zeropad =  mergedData.iloc[1, 0:sizeX]
for j in range(0, sizeX):
      zeropad[j] = 0

    
def fillOnePrevTransByColumn(row_src, row_cur, isFillZero):
    x = transcationsPerRow - (row_cur - row_src) - 1;
    sx = sizeX * x;
    ex = sizeX *(x+1)
    if(isFillZero):
        mergedData.iloc[row_cur,sx:ex] = zeropad
        #for j in range(sx, ex):
        #     mergedData.iloc[row_cur,j] = 0
    else:
         mergedData.iloc[row_cur, sx:ex] = dropcardlabel.iloc[row_src]


#####################
# for each card
#####################

cardindex=0
for i in range (1, sizeSamples):
    
    if(cardcol.card[i] == cardcol.card[cardindex]):
        i = i+1
        continue
    for j in range(cardindex, i): # j is cur row
        for k in range (0, transcationsPerRow -1):
            isFillZero =  j - (j-transcationsPerRow + k+ 1) > j - cardindex
            fillOnePrevTransByColumn( j-transcationsPerRow + k+ 1,j, isFillZero)
    
    cardindex =i
    print "convert ", i, 'of', sizeSamples
    
    
print "-------- rename idnex------"
newindex = pd.DataFrame(dropcardlabelX.head(5))
mergedIndex = pd.DataFrame(dropcardlabelX.head(5))

for i in range (1, transcationsPerRow):
    stri= '-'+ str(i);
    newindex.columns = oneTransColName + stri
    mergedIndex= pd.concat([mergedIndex, newindex], axis=1)
    
mergedData.columns = mergedIndex.columns
mergedData = pd.concat([mergedData, labelcol ], axis=1)

print mergedData.columns
#mergedData.to_csv('train-converted.csv', index=False)


listSrc= list(mergedData.columns[:]);
listDst= list(listSrc)

for i in range (0, transcationsPerRow):
    for j in range (0, sizeX):
        listDst[j*transcationsPerRow + i ] = listSrc[i*sizeX + j]
mergedDataPlus = mergedData.reindex_axis(listDst, axis=1)
print mergedDataPlus.columns
mergedDataPlus.to_csv('train-convertedPlus.csv', index=False)

end = time.clock()
print "process: %f s" % (end - start)