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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import numpy as np
#df_All = pd.read_csv("train_new.csv", sep=',')
#df_All = pd.read_csv("train_notest.csv", sep=',')
df_All = pd.read_csv("train_1108.csv", sep=',')

df_All = df_All[(df_All["label"]==0) | (df_All["label"]==1)]

df_All = df_All.fillna(-1000)


df_All = shuffle(df_All)


df_X = df_All.drop( ["certid","label"], axis=1,inplace=False)


df_y = df_All["label"]

# fit the model
clf = IsolationForest(n_estimators=1000, contamination=0.2, n_jobs=-1, bootstrap=True)
clf.fit(df_X)
# clf.fit(X_train_normal)


# predict
y_pred = clf.predict(df_X)


# change predict_labeks  (-1:1)->to (0,1)
y_pred = np.where(y_pred < 0, 0, 1)



cm1=confusion_matrix(df_y.values,y_pred)
print  cm1

print "minor class\n"
result = precision_recall_fscore_support(df_y.values,y_pred)
#print result
precision_0 = result[0][0]
recall_0 = result[1][0]
f1_0 = result[2][0]
precision_1 = result[0][1]
recall_1 = result[1][1]
f1_1 = result[2][1]
print "precision_0: ", precision_0,"  recall_0: ", recall_0, "  f1_0: ", f1_0
