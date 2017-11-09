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
import numpy as np

#df_All = pd.read_csv("train_new.csv", sep=',')
#df_All = pd.read_csv("train_notest.csv", sep=',')
df_All = pd.read_csv("train_1108.csv", sep=',')

df_All = df_All[(df_All["label"]==0) | (df_All["label"]==1)]
df_All = shuffle(df_All) 
df_All = df_All.fillna(-1)


df_X = df_All.drop( ["certid","label"], axis=1,inplace=False)

df_y = df_All["label"]


#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
fold_n=10 #n-折交叉验证：折数

kf = KFold(n_splits=fold_n)
kf.get_n_splits(df_X)  # 给出K折的折数，输出为2

df_X = np.array(df_X)
df_y = np.array(df_y)


for train_index, test_index in kf.split(df_X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = df_X[train_index], df_X[test_index]
    y_train, y_test = df_y[train_index], df_y[test_index]


    clf = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, gamma=0.01, subsample=0.7,
                        colsample_bytree=0.8, objective='binary:logistic', reg_lambda=0.01, reg_alpha=0.01, seed=133)

    clf.fit(X_train, y_train)

    pred_test = clf.predict(X_test)
    result = precision_recall_fscore_support(y_test, pred_test)
    # print result
    precision_0 = result[0][0]
    recall_0 = result[1][0]
    f1_0 = result[2][0]
    # precision_1 = result[0][1]
    # recall_1 = result[1][1]
    # f1_1 = result[2][1]
    print "precision_0: ", precision_0, "  recall_0: ", recall_0, "  f1_0: ", f1_0

#求平均
