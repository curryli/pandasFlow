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

df_All = pd.read_csv("agg_math_stable.csv", sep=',')

df_All_stat_0 = pd.read_csv("agg_cat.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_0, how='left', left_on='certid', right_on='certid')


df_All_stat = pd.read_csv("translabel_stat_2.csv", sep=',')
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

df_All_stat_8 = pd.read_csv("MCC_detail.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_8, how='left', left_on='certid', right_on='certid')

df_All_stat_9 = pd.read_csv("Mchnt_stat.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_9, how='left', left_on='certid', right_on='certid')
##########################

#########################

label_df = pd.read_csv("train_label_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)
df_All = pd.merge(left=df_All, right=label_df, how='left', left_on='certid', right_on='certid')

df_All = df_All.fillna(-1)
df_All_test = df_All[(df_All["label"] != 0) & (df_All["label"] != 1)]
df_All_T = df_All[(df_All["label"] == 0) | (df_All["label"] == 1)]

for i in range(10):
    df_All_T = shuffle(df_All_T)
    df_All_0 = df_All_T[(df_All_T["label"]==0)]
    df_All_1 = df_All_T[(df_All_T["label"]==1)]

    df_All_1 = shuffle(df_All_1)


    size_1 = df_All_1.shape[0]
    print "size1: " + str(size_1)
    split_size = size_1/4
    print "split_size: " + str(split_size)

    df_All_1_a = df_All_1.iloc[0:split_size]
    df_All_1_b = df_All_1.iloc[(split_size+1):2*split_size]
    df_All_1_c = df_All_1.iloc[(2*split_size+1):3*split_size]
    df_All_1_d = df_All_1.iloc[(3*split_size+1):]

    #####################################################
    #####################################################
    print df_All_1_a.shape[0]
    df_All_train =  pd.concat([df_All_0, df_All_1_a], axis=0)
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

    name_a = "test_a_" + str(i) + ".csv"
    np.savetxt(name_a,cerid_arr.T,delimiter=',', fmt = "%s")

    ###################################################################
    print df_All_1_b.shape[0]
    df_All_train =  pd.concat([df_All_0, df_All_1_b], axis=0)
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

    name_b = "test_b_" + str(i) + ".csv"
    np.savetxt(name_b,cerid_arr.T,delimiter=',', fmt = "%s")
    ###################################################################
    print df_All_1_c.shape[0]
    df_All_train =  pd.concat([df_All_0, df_All_1_c], axis=0)
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

    name_c = "test_c_" + str(i) + ".csv"
    np.savetxt(name_c,cerid_arr.T,delimiter=',', fmt = "%s")

    ###################################################################
    print df_All_1_d.shape[0]
    df_All_train =  pd.concat([df_All_0, df_All_1_d], axis=0)
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

    name_d = "test_d_" + str(i) + ".csv"
    np.savetxt(name_d,cerid_arr.T,delimiter=',', fmt = "%s")


