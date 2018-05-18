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
import os
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
 
from sklearn.ensemble import GradientBoostingClassifier

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


ori_cols = ["buy_freq","visit_freq","buy_interval","sv_interval","expected_time_buy","expected_time_visit","last_buy","uniq_urls","num_checkins"]

sc =StandardScaler()
df_All[ori_cols] =sc.fit_transform(df_All[ori_cols])#对数据进行标准化

cal_cols = ["is_buy_freq_NA","is_visit_freq_large","is_buy_interval_large","is_sv_interval_limit","is_sv_interval_large","is_etb_limit","is_etb_not0","is_etv_limit","is_etv_not0","is_last_buy_small","is_nc_limit"]

df_dummies = pd.get_dummies(df_All[["isbuyer","multiple_buy","multiple_visit"]])

df_X = pd.concat([df_All[ori_cols], df_All[cal_cols], df_dummies], axis=1)
used_cols = df_X.columns

df_y =  df_All['y_buy']

Alldata = pd.concat([df_X, df_y], axis=1)

seed = 7
df_T, df_V=train_test_split(Alldata, test_size=0.3)
#
df_T_0 = df_T[(df_T["y_buy"]==0)]

df_T_1 = df_T[(df_T["y_buy"]==1)]
#df_sample_0 = df_T_0.sample(n=df_T_1.shape[0])
df_sample_0 = df_T_0.sample(n=1200)

df_T = pd.concat([df_sample_0, df_T_1], axis=0)

X_train = df_T[used_cols]
y_train = df_T['y_buy']

X_test = df_V[used_cols]
y_test = df_V['y_buy']
clf = GradientBoostingClassifier(learning_rate=0.11, n_estimators=10,max_depth=4, max_features='sqrt', subsample=0.8, random_state=10, criterion="friedman_mse")

clf = clf.fit(X_train, y_train)

pred = clf.predict(X_test)

cm1=confusion_matrix(y_test, pred)
print  cm1

result = precision_recall_fscore_support(y_test,pred)
#print result
# precision_0 = result[0][0]
# recall_0 = result[1][0]
# f1_0 = result[2][0]
precision_1 = result[0][1]
recall_1 = result[1][1]
f1_1 = result[2][1]
print "precision_1: ", precision_1,"  recall_1: ", recall_1, "  f1_1: ", f1_1


#print clf.predict_proba(X_test)


##https://graphviz.gitlab.io   下载Stable Release msi  安装完以后添加环境变量  C:\Program Files (x86)\Graphviz2.38\bin


from sklearn import tree
import pydotplus

tree_list = clf.estimators_
print tree_list.shape    #(200L, 1L)  两百棵树


os.makedirs("GenedTrees/") 
for i in range(tree_list.shape[0]):
    dot_data = tree.export_graphviz(clf.estimators_[i, 0], out_file=None, feature_names=used_cols, filled=True, rounded=True)  #回归树 ，这个没用class_names=["buy", "no_buy"]
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("GenedTrees/tree_"+str(i)+".pdf")
 