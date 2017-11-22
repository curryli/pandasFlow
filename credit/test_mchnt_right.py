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

df_All = pd.read_csv("mchnt_ratio.csv", sep=',')



#df_All = df_All[["certid","mchnt_Bad_cnt","mchntcd_Bad_cnt"]]
#df_All = df_All[["certid","mchnt_risk_cnt","mchnt_Bad_cnt"]]

label_df = pd.read_csv("train_label_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)
df_All = pd.merge(left=df_All, right=label_df, how='left', left_on='certid', right_on='certid')


df_All = df_All[(df_All["label"]==0) | (df_All["label"]==1)]
df_All = df_All.fillna(-1)
df_All = shuffle(df_All)


df_X = df_All.drop( ["certid","label"], axis=1,inplace=False)

#df_X = df_X.drop( ["card_attr_cd_filled_idx-mean","card_attr_cd_filled_idx-std","card_attr_cd_filled_idx-var","trans_id_cd_filled_idx-mean","trans_id_cd_filled_idx-std","pos_entry_md_cd_filled_idx-std","pos_entry_md_cd_filled_idx-var","trans_chnl_filled_idx-std","trans_chnl_filled_idx-var","card_media_cd_filled_idx-std","card_media_cd_filled_idx-var","card_media_cd_filled_idx-std","card_media_cd_filled_idx-var","resp_cd_filled_idx-std","resp_cd_filled_idx-var","mcc_cd_filled_idx-std","mcc_cd_filled_idx-var","iss_ins_cd_filled_idx-std","iss_ins_cd_filled_idx-var"], axis=1,inplace=False)

print df_X.columns

df_y = df_All["label"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)
X_cols = X_train.columns
sc = StandardScaler()    #MinMaxScaler()    不好

#print X_train.loc[:1]

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)
clf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,gamma=0.01,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_alpha=0.1, reg_lambda=0.1,seed=27)

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
pd.DataFrame(FE_ip_tuples).to_csv("FE_IP_xgboost_drop.csv",index=True)


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
