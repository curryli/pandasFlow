# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
 
#导入随机森林算法库
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


df_All = pd.read_csv("dummy_all_withlabel.csv", header=0, sep=',')  
print df_All.shape[1]
 
 
df_X =  df_All.iloc[:, 0:-1] 
 

df_y = df_All.iloc[:, -1] 
print df_y


X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, max_features="auto",max_leaf_nodes=None, bootstrap=True)

clf = clf.fit(X_train, y_train)
  
print clf.score(X_test, y_test)

pred = clf.predict(X_test) 



precision = precision_score(y_test, pred, average="weighted") 
recall = recall_score(y_test, pred, average="weighted") 
print ("Precision:", precision) 
print ("Recall:", recall) 

confusion_matrix=confusion_matrix(y_test,pred)
print  confusion_matrix


precision = precision_score(y_test, pred, average="macro") 
recall = recall_score(y_test, pred, average="macro") 
print ("Precision:", precision) 
print ("Recall:", recall) 


#0.977356207597
#('Precision:', 0.97629591497568735)
#('Recall:', 0.97735620759690589)
#[[21608   114]
# [  410  1009]]
#('Precision:', 0.93993253475845506)
#('Recall:', 0.85290799706899123)
#[root@localhost weika]