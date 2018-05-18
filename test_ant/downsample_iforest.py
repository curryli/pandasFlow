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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

filename = r'ads_train.csv'

df_All = read_csv(filename,  sep=',', low_memory=False)

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


ori_cols = ["buy_freq","visit_freq","buy_interval","sv_interval","expected_time_buy","expected_time_visit","last_buy","multiple_buy","multiple_visit","uniq_urls","num_checkins"]

sc =StandardScaler()
df_All[ori_cols] =sc.fit_transform(df_All[ori_cols])#对数据进行标准化

cal_cols = ["is_buy_freq_NA","is_visit_freq_large","is_buy_interval_large","is_sv_interval_limit","is_sv_interval_large","is_etb_limit","is_etb_not0","is_etv_limit","is_etv_not0","is_last_buy_small","is_nc_limit"]

df_dummies = pd.get_dummies(df_All[["isbuyer"]])

df_X = pd.concat([df_All[ori_cols], df_All[cal_cols], df_dummies], axis=1)
used_cols = df_X.columns.values
#used_cols = ["num_checkins","last_buy","uniq_urls","sv_interval","expected_time_visit","visit_freq","is_last_buy_small","buy_freq","isbuyer","is_visit_freq_large","is_buy_freq_NA","is_etb_not0","buy_interval","is_sv_interval_large","multiple_buy"]

df_y =  df_All['y_buy']

Alldata = pd.concat([df_X, df_y], axis=1)

seed = 7
df_T, df_V=train_test_split(Alldata, test_size=0.3)
#



######################异常点去除
X_train = df_T[used_cols]
y_train = df_T['y_buy']

IF_clf = IsolationForest(n_estimators=200, contamination=0.05, bootstrap=True)  # n_jobs=-1,
IF_clf.fit(X_train)
y_pred_train = IF_clf.predict(X_train)

A = X_train.values
B = y_train.reshape(len(y_train),1)
C = y_pred_train.reshape(len(y_pred_train),1)
D = np.concatenate((A,B,C), axis=1)

used_cols = np.append(used_cols,"label_ori")
used_cols = np.append(used_cols,"label_IF")

new_tran_df = pd.DataFrame(D, columns = used_cols)
new_tran_df = shuffle(new_tran_df)

new_tran_df_0 = new_tran_df[new_tran_df["label_IF"] == 1]   #孤立森林的正常点

new_tran_df_a = new_tran_df_0[new_tran_df_0["label_ori"] == 1]  #孤立森林的正常点里的 buy=1 样本  删除一部分
new_tran_df_a = new_tran_df_a.sample(frac=0.7, replace=False)

new_tran_df_b = new_tran_df_0[new_tran_df_0["label_ori"] == 0]  #孤立森林的正常点里的 buy=0 样本  中等幅度降采样
new_tran_df_b = new_tran_df_b.sample(frac=0.07, replace=False)

new_tran_df_1 =  new_tran_df[new_tran_df["label_IF"] == -1]       #孤立森林的异常点

new_tran_df_c = new_tran_df_1[new_tran_df_1["label_ori"] == 1]  #孤立森林的异常点里的 buy=1 样本   保留一部分
new_tran_df_c = new_tran_df_c.sample(frac=1, replace=False)

new_tran_df_d = new_tran_df_1[new_tran_df_1["label_ori"] == 0]  #孤立森林的异常点里的 buy=0 样本  大幅度降采样
new_tran_df_d = new_tran_df_d.sample(frac=0.03, replace=False)

print new_tran_df_a.shape
print new_tran_df_b.shape
print new_tran_df_c.shape
print new_tran_df_d.shape

new_tran_df = pd.concat([new_tran_df_a, new_tran_df_b, new_tran_df_c , new_tran_df_d], axis=0)

used_cols = used_cols[:-2]
#print used_cols
X_train = new_tran_df[used_cols]
y_train = new_tran_df["label_ori"]

###################


X_test = df_V[used_cols]
y_test = df_V['y_buy']
clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, max_features="auto", max_leaf_nodes=None, bootstrap=True)

#clf = XGBClassifier(learning_rate =0.1,n_estimators=200,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)

clf = clf.fit(X_train, y_train)

FE_ip_tuples = zip(used_cols, clf.feature_importances_)
pd.DataFrame(FE_ip_tuples).to_csv("FE_ip_1.csv", index=True)



pred = clf.predict(X_test)

cm1=confusion_matrix(y_test, pred)
print  cm1

result = precision_recall_fscore_support(y_test,pred)
#print result
precision_0 = result[0][0]
recall_0 = result[1][0]
f1_0 = result[2][0]
precision_1 = result[0][1]
recall_1 = result[1][1]
f1_1 = result[2][1]
print "precision_1: ", precision_1,"  recall_1: ", recall_1, "  f1_1: ", f1_1

print classification_report(y_test, pred)

