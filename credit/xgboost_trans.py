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
from sklearn.ensemble import GradientBoostingClassifier
import datetime
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
  
df_All = pd.read_csv("new_FE_idx.csv", sep=',', low_memory=False, error_bad_lines=False)
df_All = df_All[(df_All["label"]==0) | (df_All["label"]==1)]

df_All = df_All.fillna(-1)


df_All = shuffle(df_All) 


df_X = df_All.iloc[:, 3:]

df_y = df_All["label"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)



clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

cm1=confusion_matrix(y_test,pred)
print  cm1


precision_p = float(cm1[0][0])/float((cm1[0][0] + cm1[0][1]))
recall_p = float(cm1[0][0])/float((cm1[0][0] + cm1[1][0]))
F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)

print ("Precision:", precision_p)
print ("Recall:", recall_p)
print ("F1_Score:", F1_Score)

