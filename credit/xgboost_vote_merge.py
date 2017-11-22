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

df_All = pd.read_csv("agg_math_new.csv", sep=',')

df_All_stat_0 = pd.read_csv("agg_cat.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_0, how='left', left_on='certid', right_on='certid')


df_All_stat = pd.read_csv("translabel_stat.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat, how='left', left_on='certid', right_on='certid')

df_All_stat_2 = pd.read_csv("count_label.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_2, how='left', left_on='certid', right_on='certid')

df_All_stat_3 = pd.read_csv("count_label_isnot.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_3, how='left', left_on='certid', right_on='certid')

df_All_stat_4 = pd.read_csv("groupstat_2.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_4, how='left', left_on='certid', right_on='certid')

df_All_stat_5 = pd.read_csv("addition_stat_1.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_5, how='left', left_on='certid', right_on='certid')

df_All_stat_6 = pd.read_csv("groupMCC.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_6, how='left', left_on='certid', right_on='certid')

df_All_stat_7 = pd.read_csv("addition_stat_3.csv", sep=',')  #把 addition_stat_2.csv  里面一些过拟合的标记去掉
df_All = pd.merge(left=df_All, right=df_All_stat_7, how='left', left_on='certid', right_on='certid')

# df_All_stat_8 = pd.read_csv("MCC_detail.csv", sep=',')
# df_All = pd.merge(left=df_All, right=df_All_stat_8, how='left', left_on='certid', right_on='certid')


##########################
# df_All_stat_9 = pd.read_csv("mchnt_ana.csv", sep=',')
# df_All = pd.merge(left=df_All, right=df_All_stat_9, how='left', left_on='certid', right_on='certid')
#########################

label_df = pd.read_csv("train_label_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)
df_All = pd.merge(left=df_All, right=label_df, how='left', left_on='certid', right_on='certid')

df_All = df_All.fillna(-1)


df_All_train = df_All[(df_All["label"] == 0) | (df_All["label"] == 1)]
df_All_test = df_All[(df_All["label"] != 0) & (df_All["label"] != 1)]


#####################################################
#####################################################
df_All_train = shuffle(df_All_train)
X_train = df_All_train.drop(["certid", "label"], axis=1, inplace=False)
y_train = df_All_train["label"]
clf = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, gamma=0.01, subsample=0.8, colsample_bytree=0.8,
                    objective='binary:logistic', reg_alpha=0.1, reg_lambda=0.1, seed=27)
clf = clf.fit(X_train, y_train)
X_test = df_All_test.drop(["certid", "label"], axis=1, inplace=False)
pred = clf.predict(X_test).T
cerid_arr = np.array(df_All_test["certid"]).T
cerid_arr = np.vstack((cerid_arr, pred))


for i in range(20):
    savename = "temp_nomchnt_" + str(i) + ".csv"
    print savename
    df_All_train = shuffle(df_All_train)
    X_train = df_All_train.drop(["certid", "label"], axis=1, inplace=False)
    y_train = df_All_train["label"]
    clf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,gamma=0.01,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_alpha=0.1, reg_lambda=0.1,seed=27)
    clf = clf.fit(X_train, y_train)
    X_test = df_All_test.drop(["certid", "label"], axis=1, inplace=False)
    pred = clf.predict(X_test).T
    cerid_arr = np.vstack((cerid_arr,pred))
    np.savetxt(savename, cerid_arr.T, delimiter=',', fmt="%s")

np.savetxt("result_1120_nomchnt.csv",cerid_arr.T,delimiter=',', fmt = "%s")


