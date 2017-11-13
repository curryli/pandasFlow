# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score

from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import datetime
from collections import Counter
from xgboost.sklearn import XGBClassifier
import numpy as np
from sklearn.externals import joblib

start_time = datetime.datetime.now()


################################################# 

#reader = pd.read_csv("new_FE_idx.csv", low_memory=False, iterator=True)
#reader = pd.read_csv("trans_small.csv", low_memory=False, iterator=True)
reader = pd.read_csv("cert_all_right.csv", low_memory=False, iterator=True)

loop = True
chunkSize = 100000 
chunks = []
i = 0
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
        if (i%5)==0:
            print i
        i = i+1
   
    except StopIteration:
        loop = False
        print "Iteration is stopped."
df_All = pd.concat(chunks, ignore_index=True)
print df_All.columns

#df_All = df_All.drop(["Trans_at","hist_fraud_cnt"], axis=1,inplace=False)

df_All_stat = pd.read_csv("train_1108.csv", sep=',')

#df_All_stat = df_All_stat[(df_All_stat["label"]==0) | (df_All_stat["label"]==1)]

df_All_stat= df_All_stat.drop( ["label"], axis=1,inplace=False)

df_All = pd.merge(left=df_All, right=df_All_stat, how='left', left_on='certid', right_on='certid')

df_All = shuffle(df_All)

df_All = df_All.fillna(-1)


df_All_train = df_All[(df_All["label"] == 0) | (df_All["label"] == 1)]
df_All_test = df_All[(df_All["label"] != 0) & (df_All["label"] != 1)]



X_train = df_All_train.drop(["label","certid","card_no"], axis=1,inplace=False)
y_train = df_All_train[["label"]]


np.savetxt("X_train_cols.csv",np.array(X_train.columns),fmt="%s" )


###############################################
clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)

print "start training"
clf.fit(X_train, y_train)

joblib.dump(clf, "xgboost_right.mdl")
clf = joblib.load("xgboost_right.mdl")
print "model loaded sucessfully."

###################################################
X_test = df_All_test.drop(["label", "certid", "card_no"], axis=1, inplace=False)

pred = clf.predict(X_test)

certid_test = df_All_test[["certid"]]
certid_pred = pd.DataFrame(pred,columns=["pred"])
certid_DF = pd.concat([certid_test,certid_pred], axis=1, ignore_index=True)

certid_DF.columns = ["certid","pred"]
print certid_DF.dtypes

certid_grouped = certid_DF.groupby([certid_DF['certid']])


def label_cnt(arr):  # 同一个人出现次数最多的元素
    cnt_0 = 0
    arr_values = arr.values
    for i in range(len(arr_values)):
        if arr_values[i]==float(0):
            cnt_0 = cnt_0+1
    if(cnt_0>0):
        return 0
    else:
        return 1


agg_dict = {}
agg_dict["pred"] = [label_cnt]

agg_stat_df = certid_grouped.agg(agg_dict)

agg_stat_df.columns = agg_stat_df.columns.map('{0[0]}-{0[1]}'.format)

#https://www.cnblogs.com/hhh5460/p/7067928.html
agg_stat_df.reset_index(level=0, inplace=True)
#print agg_stat_df
pred_label_DF = agg_stat_df[["certid", "pred-label_cnt"]]

pred_label_DF.to_csv("pred_label_right.csv",index=False)


end_time = datetime.datetime.now()
delta_time = str((end_time-start_time).total_seconds())
print "cost time",delta_time,"s"

