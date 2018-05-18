# -*-coding:utf-8-*-


import numpy as np
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import  set_option
#from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
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
import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier

df_Train = read_csv(r'ads_train.csv',  sep=',', low_memory=False)

df_pred = read_csv(r'ads_test.csv',  sep=',', low_memory=False)
df_pred["y_buy"] = -1

df_All = pd.concat([df_Train, df_pred], axis=0)

df_All = df_All.fillna(-1)
df_All.replace("NA",-1)

df_All["is_buy_freq_NA"] = df_All["buy_freq"].map(lambda x: 1 if x!=-1 else 0)
df_All["is_visit_freq_large"] = df_All["visit_freq"].map(lambda x: 1 if x>5 else 0)
df_All["is_buy_interval_large"] = df_All["buy_interval"].map(lambda x: 1 if x>3 else 0)
df_All["is_sv_interval_limit"] = df_All["sv_interval"].map(lambda x: 1 if x<86 else 0)
df_All["is_sv_interval_large"] = df_All["sv_interval"].map(lambda x: 1 if x>4 else 0)
df_All["is_etb_limit"] = df_All["expected_time_buy"].map(lambda x: 1 if(x>-115 and x<52) else 0)
df_All["is_etb_not0"] = df_All["expected_time_buy"].map(lambda x: 1 if x!=0 else 0)
df_All["is_etv_limit"] = df_All["expected_time_visit"].map(lambda x: 1 if x<57 else 0)
df_All["is_etv_not0"] = df_All["expected_time_visit"].map(lambda x: 1 if x!=0 else 0)
df_All["is_last_buy_small"] = df_All["last_buy"].map(lambda x: 1 if x<7 else 0)
df_All["is_nc_limit"] = df_All["num_checkins"].map(lambda x: 1 if(x>=9 and x<=6516) else 0)

ori_cols = ["buy_freq","visit_freq","buy_interval","sv_interval","expected_time_buy","expected_time_visit","last_buy","uniq_urls","num_checkins"]

sc =StandardScaler()
df_All[ori_cols] =sc.fit_transform(df_All[ori_cols])#对数据进行标准化

cal_cols = ["is_buy_freq_NA","is_visit_freq_large","is_buy_interval_large","is_sv_interval_limit","is_sv_interval_large","is_etb_limit","is_etb_not0","is_etv_limit","is_etv_not0","is_last_buy_small","is_nc_limit"]

df_dummies = pd.get_dummies(df_All[["isbuyer","multiple_buy","multiple_visit"]])

df_X = pd.concat([df_All[ori_cols], df_All[cal_cols], df_dummies], axis=1)
used_cols = df_X.columns

df_y =  df_All['y_buy']

df_All = pd.concat([df_X, df_y], axis=1)


df_pre = df_All[(df_All["y_buy"] == -1)]

#################################train################################
pred = np.zeros(df_pre.shape[0])

pred_prob = np.zeros([2, df_pre.shape[0]])

for i in range(10):
    df_T= df_All[(df_All["y_buy"] == 0) | (df_All["y_buy"] == 1)]
    df_T = shuffle(df_T)

    df_T_0 = df_T[(df_T["y_buy"]==0)]
    df_T_1 = df_T[(df_T["y_buy"]==1)]
    df_sample_0 = df_T_0.sample(n=1700)  #1200/0.7
    df_T = pd.concat([df_sample_0, df_T_1], axis=0)

    X_train = df_T[used_cols]
    y_train = df_T['y_buy']

    clf = XGBClassifier(learning_rate =0.01,n_estimators=200,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)
    clf = clf.fit(X_train, y_train)
    ##############predict
    X_pred = df_pre[used_cols]
    pred_tmp = clf.predict(X_pred).T
    pred = pred + np.array(pred_tmp)

    prob_tmp = clf.predict_proba(X_pred).T
    pred_prob = pred_prob + np.array(prob_tmp)

for i in range(10):
    df_T= df_All[(df_All["y_buy"] == 0) | (df_All["y_buy"] == 1)]
    df_T = shuffle(df_T)

    df_T_0 = df_T[(df_T["y_buy"]==0)]
    df_T_1 = df_T[(df_T["y_buy"]==1)]
    df_sample_0 = df_T_0.sample(n=1700)  #1200/0.7
    df_T = pd.concat([df_sample_0, df_T_1], axis=0)

    X_train = df_T[used_cols]
    y_train = df_T['y_buy']

    clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, max_features="auto", max_leaf_nodes=None, bootstrap=True)
    clf = clf.fit(X_train, y_train)
    ##############predict
    X_pred = df_pre[used_cols]
    pred_tmp = clf.predict(X_pred).T
    pred = pred + np.array(pred_tmp)

    prob_tmp = clf.predict_proba(X_pred).T
    pred_prob = pred_prob + np.array(prob_tmp)

pred_prob = pred_prob/20

#result = np.where(pred > 5, 1, 0)
result = np.where(pred_prob[1] > 0.5, 1, 0)
#print result

saveresult = np.vstack((pred_prob,result))

np.savetxt("saveresult.csv",saveresult.T,delimiter=',', fmt = "%s")



