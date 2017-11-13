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
from sklearn.decomposition import PCA

from sklearn.metrics import precision_recall_fscore_support

#df_All = pd.read_csv("train_new.csv", sep=',')
#df_All = pd.read_csv("train_notest.csv", sep=',')
df_All = pd.read_csv("train_1110_LS.csv", sep=',')

df_All = df_All[(df_All["label"]==0) | (df_All["label"]==1)]

df_All = df_All.fillna(-1)


df_All = shuffle(df_All)


#df_X = df_All.drop( ["certid","label","term_cd-most_frequent_item","mchnt_cd-most_frequent_item",  "aera_code",  "apply_dateNo",  "card_accprt_nm_loc-most_frequent_item"], axis=1,inplace=False)
df_X = df_All.drop( ["certid","label"], axis=1,inplace=False)

# pca = PCA(n_components = 250, svd_solver = 'full')
# #pca = PCA(n_components ='mle')
# df_X = pd.DataFrame(pca.fit_transform(df_X))

#df_X = df_X.iloc[:, 6:]

df_y = df_All["label"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)



#clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)
clf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

cm1=confusion_matrix(y_test,pred)
print  cm1

#print "Each class\n"
result = precision_recall_fscore_support(y_test,pred)
#print result
precision_0 = result[0][0]
recall_0 = result[1][0]
f1_0 = result[2][0]
precision_1 = result[0][1]
recall_1 = result[1][1]
f1_1 = result[2][1]
print "precision_0: ", precision_0,"  recall_0: ", recall_0, "  f1_0: ", f1_0
