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
import numpy as np

# df_All = pd.read_csv("train_new.csv", sep=',')
# df_All = pd.read_csv("train_notest.csv", sep=',')
df_All = pd.read_csv("train_1108.csv", sep=',')

df_All = df_All[(df_All["label"] == 0) | (df_All["label"] == 1)]

df_All = df_All.fillna(-1)

df_All = shuffle(df_All)

df_X = df_All.drop(["certid", "label"], axis=1, inplace=False)
x_columns = df_X.columns

################################################################################
df_All = shuffle(df_All)

Train_all, test_all = train_test_split(df_All, test_size=0.2)

X_test = test_all[x_columns]
y_test = test_all.label

#########################################################
train_all, valid_all = train_test_split(Train_all, test_size=0.5)
###############################################################


y_train = train_all.label
y_valid = valid_all.label

X_train = train_all[x_columns]
X_valid = valid_all[x_columns]

clf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,gamma=0.01,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=0.1,reg_alpha=0.1, seed=27)

clf = clf.fit(X_train, y_train)

print clf.score(X_valid, y_valid)

pred = clf.predict(X_valid)

print "COMPARE:::::::::::::::::::::::::::::::::::::::::::::::::"

valid_df = valid_all

# compare_idx = y_predict^y_valid  #相同为0，不同为1
compare_idx = np.bitwise_xor(pred, y_valid.values.astype(np.int32))

compare_df = pd.DataFrame(compare_idx, index=valid_df.index)
print compare_idx.shape

valid_df['compare'] = compare_df
valid_wrong = valid_df[(valid_df["compare"] == 1)]

wrong_T = valid_wrong
for i in range(0, 100):
    valid_wrong = pd.concat([valid_wrong, wrong_T], axis=0)

boosted = pd.concat([Train_all, valid_wrong], axis=0)
###############################第二次boost######################################


y_train = valid_all.label
y_valid = train_all.label

X_train = valid_all[x_columns]
X_valid = train_all[x_columns]

#clf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,gamma=0.01,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=0.1,reg_alpha=0.1, seed=27)

clf = clf.fit(X_train, y_train)

print clf.score(X_valid, y_valid)

pred = clf.predict(X_valid)

print "COMPARE:::::::::::::::::::::::::::::::::::::::::::::::::"

valid_df = valid_all

# compare_idx = y_predict^y_valid  #相同为0，不同为1

print pred
print y_valid.values.astype(np.int32)
compare_idx = np.bitwise_xor(pred, y_valid.values.astype(np.int32))

compare_df = pd.DataFrame(compare_idx, index=valid_df.index)
print compare_idx.shape

valid_df['compare'] = compare_df
valid_wrong = valid_df[(valid_df["compare"] == 1)]

wrong_T = valid_wrong
for i in range(0, 100):
    valid_wrong = pd.concat([valid_wrong, wrong_T], axis=0)

boosted = pd.concat([boosted, valid_wrong], axis=0)

###############################################################################


boosted = shuffle(boosted)

X_boosted = boosted[x_columns]
y_boosted = boosted.label

#######################################




clf = clf.fit(X_boosted, y_boosted)

print clf.score(X_test, y_test)

pred = clf.predict(X_test)

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
