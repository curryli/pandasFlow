# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score

# 导入随机森林算法库
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from xgboost.sklearn import XGBClassifier


df_All = pd.read_csv("train.csv", sep=',')
df_All = df_All.fillna(-1)

df_All_train = df_All[(df_All["label"] == 0) | (df_All["label"] == 1)]
df_All_test = df_All[(df_All["label"] != 0) & (df_All["label"] != 1)]

print df_All_test.size




df_All_train = shuffle(df_All_train)

X_train = df_All_train.drop(["certid", "label"], axis=1, inplace=False)

y_train = df_All_train["label"]

clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)

# clf = GradientBoostingClassifier(random_state=100, n_estimators=100)
clf = clf.fit(X_train, y_train)


joblib.dump(clf, "xgboost.mdl")

clf = joblib.load("xgboost.mdl")
print "model loaded sucessfully."


X_test = df_All_test.drop(["certid", "label"], axis=1, inplace=False)

pred = clf.predict(X_test).T

print pred.shape

cerid_arr = np.array(df_All_test["certid"]).T

result = np.vstack((cerid_arr,pred))
print np.savetxt("xgboost_results.csv",result.T,delimiter=',', fmt = "%s")

