# -*- coding: utf-8 -*-
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
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# df_All = pd.read_csv("train_new.csv", sep=',')
# df_All = pd.read_csv("train_notest.csv", sep=',')
df_All = pd.read_csv("train_1108.csv", sep=',')

df_All = df_All[(df_All["label"] == 0) | (df_All["label"] == 1)]

df_All = df_All.fillna(-1)

df_All = shuffle(df_All)

df_X = df_All.drop(["certid", "label"], axis=1, inplace=False)

df_y = df_All["label"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)
X_cols = list(X_train.columns.values)
y_train = y_train.values
y_test = y_test.values


# IF_clf = LocalOutlierFactor(contamination=0.1)
# y_pred_train=IF_clf.fit_predict(X_train)
IF_clf = IsolationForest(n_estimators=1000, contamination=0.2, n_jobs=-1, bootstrap=True)
IF_clf.fit(X_train)
y_pred_train = IF_clf.predict(X_train)

A = X_train.values
B = y_train.reshape(len(y_train),1)
C = y_pred_train.reshape(len(y_pred_train),1)
# print A.shape
# print B.shape
# print C.shape

D = np.concatenate((A,B,C), axis=1)
#D = np.concatenate((D,C), axis=1)

cols = X_cols

cols.append("label_ori")
cols.append("label_IF")



new_tran_df = pd.DataFrame(D, columns = cols)
print new_tran_df.shape

new_tran_df["label_IF"].to_csv("label_IF.csv")
# new_tran_df = new_tran_df[new_tran_df["label_IF"]>0]

new_tran_df_0 = new_tran_df[new_tran_df["label_IF"] == 1]   #孤立森林的正常点
print new_tran_df_0.shape
new_tran_df_a = new_tran_df_0[new_tran_df_0["label_ori"] == 0]  #孤立森林的正常点里的违约样本
new_tran_df_a = new_tran_df_a.sample(frac=1, replace=False)
print new_tran_df_a.shape

new_tran_df_b = new_tran_df_0[new_tran_df_0["label_ori"] == 1]  #孤立森林的正常点里的正常样本
new_tran_df_b = new_tran_df_b.sample(frac=0.1, replace=False)
print new_tran_df_b.shape


new_tran_df_1 =  new_tran_df[new_tran_df["label_IF"] == -1]       #孤立森林的异常点
print new_tran_df_1.shape

new_tran_df_c = new_tran_df_1[new_tran_df_1["label_ori"] == 0]  #孤立森林的异常点里的违约样本
new_tran_df_c = new_tran_df_c.sample(frac=0.01, replace=False)
print new_tran_df_c.shape

new_tran_df_d = new_tran_df_1[new_tran_df_1["label_ori"] == 1]  #孤立森林的异常点里的正常样本
new_tran_df_d = new_tran_df_d.sample(frac=0.01, replace=False)
print new_tran_df_d.shape


new_tran_df = pd.concat([new_tran_df_a, new_tran_df_b, new_tran_df_c , new_tran_df_d], axis=0)   #导致recall很大


#new_tran_df =  new_tran_df_0.append(new_tran_df_1)

used_cols = X_cols[:-2]
#print used_cols
X_train = new_tran_df[used_cols]
y_train = new_tran_df["label_ori"]

# clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)
clf = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, gamma=0.01, subsample=0.8, colsample_bytree=0.8,
                    objective='binary:logistic', reg_lambda=0.1, reg_alpha=0.1, seed=27)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

cm1 = confusion_matrix(y_test, pred)
print  cm1

precision_p = float(cm1[0][0]) / float((cm1[0][0] + cm1[1][0]))
recall_p = float(cm1[0][0]) / float((cm1[0][0] + cm1[0][1]))
F1_Score = 2 * precision_p * recall_p / (precision_p + recall_p)

print ("Precision:", precision_p)
print ("Recall:", recall_p)
print ("F1_Score:", F1_Score)

FE_ip_tuples = zip(X_cols, clf.feature_importances_)
pd.DataFrame(FE_ip_tuples).to_csv("FE_ip_xgboost_1108_2.csv", index=True)

# Compute precision, recall, F-measure and support for each class
# print "weighted\n"
# print precision_recall_fscore_support(y_test,pred, average='weighted')

print "Each class\n"
result = precision_recall_fscore_support(y_test, pred)
# print result
precision_0 = result[0][0]
recall_0 = result[1][0]
f1_0 = result[2][0]
precision_1 = result[0][1]
recall_1 = result[1][1]
f1_1 = result[2][1]
print "precision_0: ", precision_0, "  recall_0: ", recall_0, "  f1_0: ", f1_0
