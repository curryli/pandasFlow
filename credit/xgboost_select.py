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

df_All = df_All.fillna(-1)


#df_All = shuffle(df_All)


df_X = df_All.drop( ["certid","label"], axis=1,inplace=False)



df_All_xyk = pd.read_csv("train_1109_xyk.csv", sep=',')
df_All_xyk = df_All_xyk[(df_All_xyk["label_xyk"]==0) | (df_All_xyk["label_xyk"]==1)]

df_X_xyk = df_All_xyk.drop( ["certid","label_xyk"], axis=1,inplace=False)
df_X_xyk = df_X_xyk.fillna(-1)


# A = df_X.values
# B = df_X_xyk.values
# print A.shape
# print B.shape
# df_X = np.concatenate((A,B), axis=1)

df_X = pd.concat([df_X, df_X_xyk], axis=1)




print df_X.shape

df_y = df_All["label"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)
X_cols = X_train.columns
sc = StandardScaler()    #MinMaxScaler()    不好

#print X_train.loc[:1]

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)
clf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,gamma=0.01,subsample=0.7,colsample_bytree=0.4,objective= 'binary:logistic', reg_lambda=0.01,reg_alpha=0.01,seed=133)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

cm1=confusion_matrix(y_test,pred)
print  cm1

precision_p = float(cm1[0][0])/float((cm1[0][0] + cm1[1][0]))
recall_p = float(cm1[0][0])/float((cm1[0][0] + cm1[0][1]))
F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)

print ("Precision:", precision_p)
print ("Recall:", recall_p)
print ("F1_Score:", F1_Score)

FE_ip_tuples = zip(X_cols, clf.feature_importances_)
pd.DataFrame(FE_ip_tuples).to_csv("FE_ip_xgboost_1109_xyk.csv",index=True)


#Compute precision, recall, F-measure and support for each class
# print "weighted\n"
# print precision_recall_fscore_support(y_test,pred, average='weighted')

print "Each class\n"
result = precision_recall_fscore_support(y_test,pred)
#print result
precision_0 = result[0][0]
recall_0 = result[1][0]
f1_0 = result[2][0]
precision_1 = result[0][1]
recall_1 = result[1][1]
f1_1 = result[2][1]
print "precision_0: ", precision_0,"  recall_0: ", recall_0, "  f1_0: ", f1_0
