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


df_All = pd.read_csv("train.csv", sep=',')
df_All = df_All.fillna(-1)

df_All_train = df_All[(df_All["label"] == 0) | (df_All["label"] == 1)]
df_All_test = df_All[(df_All["label"] != 0) & (df_All["label"] != 1)]

print df_All_test.size
#
#
#
#
# df_All_train = shuffle(df_All_train)
#
# X_train = df_All_train.drop(["idx", "certid", "label"], axis=1, inplace=False)
#
# y_train = df_All_train["label"]
#
# # n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
# clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=500, max_depth=10, min_samples_leaf=60,
#                                  min_samples_split=1200, max_features=10, subsample=0.7, random_state=10)
# # clf = GradientBoostingClassifier(random_state=100, n_estimators=100)
# clf = clf.fit(X_train, y_train)
#
#
# joblib.dump(clf, "gbdt.mdl")

clf = joblib.load("gbdt.mdl")
print "model loaded sucessfully."


X_test = df_All_test.drop(["certid", "label"], axis=1, inplace=False)

pred = clf.predict(X_test).T

print pred.shape

cerid_arr = np.array(df_All_test["certid"]).T

result = np.vstack((cerid_arr,pred))
print np.savetxt("2.csv",result.T,delimiter=',', fmt = "%s")

