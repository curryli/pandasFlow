# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
 
#导入随机森林算法库
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle 
 
  
df_All = pd.read_csv("LSTM_converted_3.csv", sep=',') 
#df_All = pd.read_csv("idx_new_08_del.csv", sep=',') 
df_All = shuffle(df_All) 
print df_All.shape[1]
  
df_X = df_All.iloc[:, :-2]
#df_X = df_All.iloc[:, 1:-2]
#df_X = df_All[["hour","trans_at","total_disc_at"]]
  

df_y = df_All["label"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)


#n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features="auto",max_leaf_nodes=None, bootstrap=True)

clf = clf.fit(X_train, y_train)
  
print clf.score(X_test, y_test)

pred = clf.predict(X_test) 

confusion_matrix=confusion_matrix(y_test,pred)
print  confusion_matrix

precision_p = float(confusion_matrix[1][1])/float((confusion_matrix[0][1] + confusion_matrix[1][1]))
recall_p = float(confusion_matrix[1][1])/float((confusion_matrix[1][0] + confusion_matrix[1][1]))
 
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 


#0.990622704291  10棵树
#[[21721    84]
# [  133  1203]]
#('Precision:', 0.96432304616161124)
#('Recall:', 0.94829838717428705)
#('Precision:', 0.99049738866427095)
#('Recall:', 0.99062270429108512)

#100棵树
#[[21621   109]
# [  100  1311]]
 