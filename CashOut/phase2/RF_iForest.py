# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:37:55 2017

@author: Administrator
"""
import pandas as pd 
import numpy as np

from pandas import DataFrame
from imblearn.over_sampling import SMOTE
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import shuffle  
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

label='label' # 
cardcol= 'pri_acct_no_conv'
 
def data_prepration(x): 
#    x_features= x.ix[:,x.columns !="label"]
#    x_labels=x.ix[:,x.columns=="label"]
    x_columns = [colname for colname in x.columns if colname not in [label,cardcol]]
    x_features= x[x_columns] 
    x_labels=x.ix[:,x.columns=="label"]          
    x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=0.2)
    print("length of training data")
    print(len(x_features_train))
    print("length of test data")
    print(len(x_features_test))
    return(x_features_train,x_features_test,x_labels_train,x_labels_test)

df = pd.read_csv("idx_weika_07.csv")

df = shuffle(df) 
data_train_X,data_test_X,data_train_y,data_test_y=data_prepration(df)
print pd.value_counts(data_test_y['label'])

ilf = IsolationForest(n_estimators=100,
                  n_jobs=-1,          
                  verbose=2,
                  contamination=0.1
)
# 训练
ilf.fit(data_train_X)
shape = data_train_X.shape[0]
batch = 10**3

all_pred = []
for i in range(shape/batch+1):
    start = i * batch
    end = (i+1) * batch
    test = data_train_X[start:end]
# 预测
    pred = ilf.predict(test)
    all_pred.extend(pred)
    
outlier=DataFrame(all_pred)

data_o_X=data_train_X[outlier.values==-1]
data_o_y=data_train_y[outlier.values==-1]

print 'ioslated trans in fraud:%d' %(np.count_nonzero(data_train_y[data_train_y.index.isin(data_o_y.index)]))
print 'fraud num in training sample:%d' %data_train_y[data_train_y['label']==1].shape[0]

#oversample 
for i in range(0,3):
#    print i
    data_train_X=pd.concat([data_train_X,data_o_X], axis = 0)
    data_train_y=pd.concat([data_train_y,data_o_y], axis = 0)

print "start training:"
#随机森林
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=10, max_features="auto",max_leaf_nodes=None, bootstrap=True)
 
#sampleWeigth=np.random.rand(data_train_y.shape[0])
#clf = clf.fit(data_train_X, data_train_y,sampleWeigth) 
clf = clf.fit(data_train_X, data_train_y)
  
print clf.score(data_test_X, data_test_y)

pred = clf.predict(data_test_X) 

temp_m=confusion_matrix(data_test_y,pred)
print temp_m   
precision_p = float(temp_m[1][1])/float((temp_m[0][1] + temp_m[1][1]))
recall_p = float(temp_m[1][1])/float((temp_m[1][0] + temp_m[1][1]))
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 