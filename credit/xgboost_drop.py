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

from sklearn.metrics import precision_recall_fscore_support

#df_All = pd.read_csv("train_new.csv", sep=',')
#df_All = pd.read_csv("train_notest.csv", sep=',')
df_All = pd.read_csv("train_shuffled.csv", sep=',')

df_All = df_All[(df_All["label"]==0) | (df_All["label"]==1)]

df_All = df_All.fillna(-1)


df_All = shuffle(df_All)


df_X = df_All.drop( ["certid","label"], axis=1,inplace=False)
df_X = df_X[["aera_code","apply_dateNo","card_accprt_nm_loc-most_frequent_item","mchnt_cd-most_frequent_item","iss_ins_cd-most_frequent_item","city","dateNo-std","dateNo-mean","apply_mean_delta","tras_at_max_mean_ratio","county","dateNo-median","resp_cd-most_frequent_cnt","dateNo-min","term_cd-most_frequent_item","date-most_frequent_item","dateNo-max","age","apply_max_delta","aera_code_encode","hour-most_frequent_item","auth_id_resp_cd-countDistinct","rcv_settle_at-median","dateNo-sum","min_apply_delta","dateNo-peak_to_peak","rcv_settle_at-mean","mcc_cd-most_frequent_item","month-most_frequent_cnt","term_cd-most_frequent_cnt","rcv_settle_at-std","Trans_at-min","rcv_settle_at-min","mcc_cd-most_frequent_cnt","month-most_frequent_item","weekday-most_frequent_item","is_risk_term_cd-sum","month-countDistinct","mchnt_cd-most_frequent_cnt","Trans_at-mean","term_cd-countDistinct","card_accprt_nm_loc-most_frequent_cnt","rcv_settle_at-sum","card_accprt_nm_loc-countDistinct","Trans_at-median","mcc_cd-countDistinct","is_risk_auth_id_resp_cd-sum","auth_id_resp_cd-most_frequent_cnt","rcv_settle_at-peak_to_peak","date-most_frequent_cnt","hour-countDistinct","mchnt_cd-countDistinct","is_risk_aera_code","iss_ins_cd-most_frequent_cnt","resp_cd-countDistinct","rcv_settle_at-max","dateNo-var","trans_id_cd-countDistinct","card_media_cd-most_frequent_cnt","trans_chnl-most_frequent_cnt","card_no-countDistinct","Trans_at-sum","card_attr_cd-most_frequent_cnt","hour-most_frequent_cnt","trans_id_cd-most_frequent_cnt","stageInMonth-most_frequent_cnt","date-countDistinct","pos_entry_md_cd-most_frequent_cnt","resp_cd-most_frequent_item","Trans_at-std","pos_entry_md_cd-countDistinct","pos_cond_cd-most_frequent_cnt","weekday-most_frequent_cnt","trans_id_cd-most_frequent_item","stageInMonth-most_frequent_item","fwd_settle_at-mean","is_risk_card_accprt_nm_loc-sum","pos_entry_md_cd-most_frequent_item","Trans_at-peak_to_peak","Trans_at-max","sex_female","fwd_settle_at-min","card_media_cd-countDistinct","iss_ins_cd-countDistinct","rcv_settle_at-var","orig_trans_st-most_frequent_cnt","card_attr_cd-countDistinct","trans_chnl-countDistinct","fwd_settle_conv_rt-most_frequent_cnt","trans_chnl-most_frequent_item","card_media_cd-most_frequent_item","fwd_settle_at-std","pos_cond_cd-countDistinct","fwd_settle_at-sum","fwd_settle_at-median","is_risk_rcv_settle_conv_rt-sum","is_risk_pos_cond_cd-sum","is_risk_iss_ins_cd-sum","orig_trans_st-countDistinct","weekday-countDistinct","auth_id_resp_cd-most_frequent_item","fwd_settle_at-max","is_risk_mchnt_cd-sum"]]

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
