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
 
from sklearn.model_selection import cross_val_score, GridSearchCV  #using Kfold and if needed, GridSearch object in analysis
from sklearn.utils import shuffle                                  # shuffling our own made dataset
 
from sklearn.model_selection import train_test_split
 
from sklearn.svm import SVC
 
import time

labelName="label"
cardname="pri_acct_no_conv"

global timeStepPerRow
timeStepPerRow = 5

# minTimeStepNeeded must >=1 AND <= timeStepPerRow
minTimeStepNeeded = 1

start = time.clock()

allsamples = pd.read_csv('weika_GBDT_07.csv')
sizeSamples= allsamples.shape[0]

## sort allsamples by pri_acct_no_conv/time if needed
print "Sorting according to pri_acct_no_conv and time ..."
allsamples = allsamples.sort_values(by=[cardname, 'tfr_dt_tm'], inplace=False)#sort_index
allsamples = allsamples.reset_index(drop = True)
## end sort
print "sort done"

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
for i in range (1, timeStepPerRow):
    mergedData= pd.concat([mergedData, dropcardlabelX], axis=1)
 
#mergedData.to_csv('train_s1.csv', index=False)
    
    
global zeropad
zeropad =  mergedData.iloc[1, 0:sizeX]
for j in range(0, sizeX):
      zeropad[j] = 0

# count 0, pick  zeroCountPerRow< timeStepPerRow- minTimeStep
global zeroCountPerRow 
zeroCountPerRow= np.zeros (sizeSamples )
    
def fillOnePrevTransByColumn(row_src, row_cur, isFillZero):
    x = timeStepPerRow - (row_cur - row_src) - 1;
    sx = sizeX * x;
    ex = sizeX *(x+1)
    if(isFillZero):
        mergedData.iloc[row_cur,sx:ex] = zeropad
        zeroCountPerRow[row_cur] = zeroCountPerRow[row_cur]+1
        #for j in range(sx, ex):
        #     mergedData.iloc[row_cur,j] = 0
    else:
        mergedData.iloc[row_cur, sx:ex] = dropcardlabel.iloc[row_src]


#####################
# for each pri_acct_no_conv
#####################

#for k in range (0, timeStepPerRow -1):
#    fillOnePrevTransByColumn( 0, 0, True)


print "drop done, start converting"

cardindex=0
for i in range (1, sizeSamples):
    print "convert ", i, 'of', cardindex
    lastCol = i
    if( i< sizeSamples-1):
        if(cardcol.pri_acct_no_conv[i] == cardcol.pri_acct_no_conv[cardindex]):
            i = i+1
            continue
    else:
        if(i<= cardindex):
            continue
        else:
            lastCol = sizeSamples

    for j in range(cardindex, lastCol): # j is cur row
        for k in range (0, timeStepPerRow -1):
            isFillZero =  j - (j-timeStepPerRow + k+ 1) > j - cardindex
            fillOnePrevTransByColumn( j-timeStepPerRow + k+ 1,j, isFillZero)
    
    
    cardindex =i

    
print "-------- rename idnex------"
newindex = pd.DataFrame(dropcardlabelX.head(5))
mergedIndex = pd.DataFrame(dropcardlabelX.head(5))

for i in range (1, timeStepPerRow):
    stri= '-'+ str(i);
    newindex.columns = oneTransColName + stri
    mergedIndex= pd.concat([mergedIndex, newindex], axis=1)
    
mergedData.columns = mergedIndex.columns
mergedData = pd.concat([cardcol, mergedData, labelcol ], axis=1)

mergedDataClipped = mergedData[zeroCountPerRow< timeStepPerRow - minTimeStepNeeded+1]
print mergedDataClipped.columns
mergedDataClipped.to_csv('convert_5_weika_GBDT_07.csv', index=False)

end = time.clock()
print "process: %f s" % (end - start)