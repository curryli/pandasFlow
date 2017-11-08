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
  
df_All = pd.read_csv("train.csv", sep=',')
df_All = df_All[(df_All["label"]==0) | (df_All["label"]==1)]

df_All = df_All.fillna(-1)


df_All = shuffle(df_All) 


df_X = df_All.drop( ["certid","label"], axis=1,inplace=False)
df_X = df_X[["aera_code","apply_dateNo","card_accprt_nm_loc-most_frequent_item","mchnt_cd-most_frequent_item","iss_ins_cd-most_frequent_item","dateNo-std","dateNo-mean","apply_mean_delta","term_cd-most_frequent_item","city","resp_cd-most_frequent_cnt","dateNo-median","date-most_frequent_item","age","dateNo-max","apply_max_delta","min_apply_delta","hour-most_frequent_item","dateNo-sum","aera_code_encode","dateNo-min","auth_id_resp_cd-countDistinct","dateNo-peak_to_peak","mcc_cd-most_frequent_item","mcc_cd-most_frequent_cnt","month-most_frequent_cnt","term_cd-most_frequent_cnt","auth_id_resp_cd-most_frequent_cnt","card_accprt_nm_loc-most_frequent_cnt","hour-countDistinct","month-most_frequent_item","mchnt_cd-countDistinct","iss_ins_cd-most_frequent_cnt","term_cd-countDistinct","card_accprt_nm_loc-countDistinct","mcc_cd-countDistinct","mchnt_cd-most_frequent_cnt","weekday-most_frequent_item","month-countDistinct"]]

df_y = df_All["label"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)



clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

cm1=confusion_matrix(y_test,pred)
print  cm1


precision_p = float(cm1[0][0])/float((cm1[0][0] + cm1[0][1]))
recall_p = float(cm1[0][0])/float((cm1[0][0] + cm1[1][0]))
F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)

print ("Precision:", precision_p)
print ("Recall:", recall_p)
print ("F1_Score:", F1_Score)

