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
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
  
df_All = pd.read_csv("train.csv", sep=',')
df_All = df_All[(df_All["label"]==0) | (df_All["label"]==1)]


df_All = df_All.fillna(-1)
df_All = shuffle(df_All) 

df_y = df_All["label"]
df_X = df_All.drop("label",axis=1,  inplace=False)

df_X = pd.get_dummies(df_X)

print "dummy done.", df_X.shape

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)

clf = XGBClassifier(learning_rate =0.1,n_estimators=3000,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)

clf = clf.fit(X_train, y_train)

# print clf.score(X_test, y_test)

pred = clf.predict(X_test)

cm1 = confusion_matrix(y_test, pred)
print  cm1

precision_p = float(cm1[0][0]) / float((cm1[0][0] + cm1[0][1]))
recall_p = float(cm1[0][0]) / float((cm1[0][0] + cm1[1][0]))
F1_Score = 2 * precision_p * recall_p / (precision_p + recall_p)

print ("Precision:", precision_p)
print ("Recall:", recall_p)
print ("F1_Score:", F1_Score)

FE_ip_tuples = zip(X_train.columns, clf.feature_importances_)
pd.DataFrame(FE_ip_tuples).to_csv("FE_ip_tuples_dummy.csv", index=True)
 