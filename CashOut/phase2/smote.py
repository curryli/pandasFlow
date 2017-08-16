# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 12:21:33 2017
conda install -c glemaitre imbalanced-learn 
或者pip install -U imbalanced-learn
https://github.com/scikit-learn-contrib/imbalanced-learn
@author: Administrator
"""
import pandas as pd 
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import shuffle  

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

df = pd.read_csv("idx_new_08_del.csv")

for i in range(0,10):
    df = shuffle(df) 
    data_train_X,data_test_X,data_train_y,data_test_y=data_prepration(df)
    print pd.value_counts(data_test_y['label'])
    #调用smote
    smote = SMOTE(random_state=0) 
    os_data_X,os_data_y=smote.fit_sample(data_train_X.values,data_train_y.values.ravel())
    
    print np.count_nonzero(os_data_y)
    
    print "start training:"
    #随机森林
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=10, max_features="auto",max_leaf_nodes=None, bootstrap=True)
    clf = clf.fit(os_data_X, os_data_y)
      
    print clf.score(data_test_X, data_test_y)
    
    pred = clf.predict(data_test_X) 
    temp_m=confusion_matrix(data_test_y,pred)
    
    print temp_m   
    precision_p = float(temp_m[1][1])/float((temp_m[0][1] + temp_m[1][1]))
    recall_p = float(temp_m[1][1])/float((temp_m[1][0] + temp_m[1][1]))
    print ("Precision:", precision_p) 
    print ("Recall:", recall_p) 