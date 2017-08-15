# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time 
from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold  
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import shuffle  
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv("idx_new_08_del.csv")
label='label' # label的值就是二元分类的输出
cardcol= 'pri_acct_no_conv'
 
df = shuffle(df) 
total_size = df.shape[0]  

#df_T = df.iloc[: 4*(total_size/5)] 
#df_V = df.iloc[4*(total_size/5)+1:] 
df_T,df_V=train_test_split(df, test_size=0.2)
print "total_size: " ,total_size, "df_T size: ", df_T.shape[0],"df_V size: ",df_V.shape[0]

#采样
fraud_size = 1
normal_size = 1

Fraud = df_T[df.label == 1]
Fraud_train=Fraud
for i in range(0,fraud_size-1):
    Fraud_train=pd.concat([Fraud_train,Fraud], axis = 0)

normal = df_T[df.label == 0]
normal_train=normal
for i in range(0,normal_size-1):
    normal_train=pd.concat([normal_train,normal], axis = 0)

print "Fraud_train_o size", Fraud.shape[0], "normal_train_o size",normal.shape[0]
print "Fraud_train size", Fraud_train.shape[0], "normal_train size",normal_train.shape[0]

df_train = pd.concat([Fraud_train,normal_train], axis = 0)
 
 

x_columns = [x for x in df_train.columns if x not in [label,cardcol]]
X_train = df_train[x_columns]
y_train = df_train[label]

#print "datasets size:%d" % X_train.shape[0]


X_test = df_V[x_columns]
y_test = df_V[label]

print "start training:"

#clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=10, max_features="auto",max_leaf_nodes=None, bootstrap=True)
#clf = clf.fit(X_train, y_train)
#  
#print clf.score(X_test, y_test)
#
#pred = clf.predict(X_test) 
#clf = RandomForestClassifier(oob_score=True, random_state=10)
start=time.time()
bdt = AdaBoostClassifier(RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=10, max_features="auto",max_leaf_nodes=None, bootstrap=True),
                         algorithm="SAMME",
                         n_estimators=10, learning_rate=0.1)
#bdt.fit(X_train, y_train)
end1=time.time()
print "create model time used: %f" % (end1-start)

bdt = bdt.fit(X_train, y_train)
  
end2=time.time()
print "create model time used: %f"  %(end2-end1)
print bdt.score(X_test, y_test)

pred = bdt.predict(X_test) 

confusion_matrix=confusion_matrix(y_test,pred)
print  confusion_matrix


precision_p = float(confusion_matrix[1][1])/float((confusion_matrix[0][1] + confusion_matrix[1][1]))
recall_p = float(confusion_matrix[1][1])/float((confusion_matrix[1][0] + confusion_matrix[1][1]))
 
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 