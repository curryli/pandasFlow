# -*- coding: utf-8 -*-
"""
Function: Prepare data-split mixed data to fraud/clean data for both train/test
Input: train.csv/test.csv
Output:train_fraud.csv/train_clean.csv and test_fraud.csv/test_clean.csv
@author: yuwei
"""

import pandas as pd                #data analysing tool for python
import numpy as np
from IPython.display import display
from sklearn.preprocessing import StandardScaler             
import warnings
warnings.filterwarnings('ignore')

#import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/")

labelName ="label"
cardName = "card"
isDropLabel = True
isSC = True #False

########################
# train file
########################
print ("loading train.csv ...")
train_data = pd.read_csv("train.csv")
sc = StandardScaler()
if(isSC):
    train_data_nolabel=train_data.drop(labelName, axis = 1)
    train_data_nolabel=train_data_nolabel.drop(cardName, axis = 1)
    train_data_nolabel_np=sc.fit_transform(train_data_nolabel)

print ("writing train_fraud.csv ...")
train_fraud= train_data[train_data[labelName]==1]
if isDropLabel:
    train_fraud = train_fraud.drop(labelName, axis = 1)
    if(isSC):
        train_fraud = train_fraud.drop(cardName, axis=1)
        train_fraud_np = sc.transform(train_fraud)
        train_fraud = pd.DataFrame(train_fraud_np)
        train_fraud.columns = train_data_nolabel.columns
train_fraud.to_csv('train_fraud.csv')
print "1/4: Train fraud count=", train_fraud.shape[0], "/", train_data.shape[0]

print ("writing train_clean.csv ...")
train_clean= train_data[train_data[labelName]==0]
if isDropLabel:
    train_clean = train_clean.drop(labelName, axis = 1)
    if(isSC):
        train_clean = train_clean.drop(cardName, axis=1)
        train_clean_np = sc.transform(train_clean)
        train_clean = pd.DataFrame(train_clean_np)
        train_clean.columns = train_data_nolabel.columns
train_clean.to_csv('train_clean.csv')
print "2/4: Train clean count=", train_clean.shape[0],"/", train_data.shape[0]

########################
# test file
########################
print ("\nloading test.csv ...")
test_data  = pd.read_csv("test.csv")

print ("writing test_fraud.csv ...")
test_fraud= test_data[test_data[labelName]==1]
if isDropLabel:
    test_fraud = test_fraud.drop(labelName, axis = 1)
    if(isSC):
        test_fraud=test_fraud.drop(cardName, axis=1)
        test_fraud_np = sc.transform(test_fraud)
        test_fraud = pd.DataFrame(test_fraud_np)
        test_fraud.columns = train_data_nolabel.columns
test_fraud.to_csv('test_fraud.csv')
print "3/4: Test fraud count=", test_fraud.shape[0], "/", test_data.shape[0]

print ("writing test_clean.csv ...")
test_clean= test_data[test_data[labelName]==0]
if isDropLabel:
    test_clean = test_clean.drop(labelName, axis = 1)
    if(isSC):
        test_clean=test_clean.drop(cardName, axis=1)
        test_clean_np = sc.transform(test_clean)
        test_clean = pd.DataFrame(test_clean_np)
        test_clean.columns = train_data_nolabel.columns
test_clean.to_csv('test_clean.csv')
print "4/4: Test clean count=", test_clean.shape[0], "/", test_data.shape[0]




