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

fraudname = r'fraud.csv'
normalname = r'normal.csv'

df_fraud = read_csv(fraudname,  sep=',', dtype=str)
df_normal = read_csv(normalname,  sep=',',  dtype=str)

df_fraud["label"] = 1
df_normal["label"] = 0

df_All = pd.concat([df_fraud,df_normal], axis = 0)
df_All = df_All.fillna(-1)

df_All = shuffle(df_All)

sus_cols = ["trans_at", "settle_at"]
df_All["trans_at"] = df_All["trans_at"].astype(np.double)
df_All["settle_at"] = df_All["settle_at"].astype(np.double)

dis_cols = ["resp_cd","app_ins_inf","acq_ins_id_cd","mchnt_tp","card_attr","acct_class","app_ins_id_cd","fwd_ins_id_cd","trans_curr_cd","trans_tp","proc_st","ins_pay_mode","up_discount","app_discount","ctrl_rule1","mer_version","app_version","order_type","app_ntf_st","acq_ntf_st","proc_sys","mchnt_back_url","app_back_url","mer_cert_id","mchnt_nm","acq_ins_inf","country_cd","area_cd"]
df_dummies = pd.get_dummies(df_All[dis_cols])
df_X = pd.concat([df_All[sus_cols],df_dummies], axis=1)
used_cols = df_X.columns


sc =StandardScaler()
df_X =sc.fit_transform(df_X)#对数据进行标准化

df_y = df_All["label"]


X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)

#n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features="auto",max_leaf_nodes=None, bootstrap=True)

clf = clf.fit(X_train, y_train)


FE_ip_tuples = zip(used_cols, clf.feature_importances_)
pd.DataFrame(FE_ip_tuples).to_csv("FE_ip.csv", index=True)



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

