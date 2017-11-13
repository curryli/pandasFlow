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

df_All = df_All[(df_All["label"] == 0) | (df_All["label"] == 1)]


df_All_stat = pd.read_csv("train_1108.csv", sep=',')

df_All_stat = df_All_stat[(df_All_stat["label"]==0) | (df_All_stat["label"]==1)]

df_All_stat= df_All_stat.drop( ["label"], axis=1,inplace=False)

df_All = pd.merge(left=df_All, right=df_All_stat, how='left', left_on='certid', right_on='certid')



df_All = shuffle(df_All)

df_All = df_All.fillna(-1)

df_X = df_All.drop(["label","certid","card_no"], axis=1,inplace=False)



df_y = df_All[["certid","label"]]



X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)

np.savetxt("X_train_cols.csv",np.array(X_train.columns),fmt="%s" )
###############################################

certid_test = y_test

y_train = y_train.drop(["certid"], axis=1,inplace=False)
y_test = y_test.drop(["certid"], axis=1,inplace=False)

clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)

print "start training"
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

cm1=confusion_matrix(y_test,pred)
print  cm1



print "For Trans:\n"
result = precision_recall_fscore_support(y_test,pred)
#print result
precision_0 = result[0][0]
recall_0 = result[1][0]
f1_0 = result[2][0]
precision_1 = result[0][1]
recall_1 = result[1][1]
f1_1 = result[2][1]
print "precision_0: ", precision_0,"  recall_0: ", recall_0, "  f1_0: ", f1_0



#print "certid_test_ori\n",certid_test
certid_test.index = range(certid_test.shape[0])
#print "certid_test\n",certid_test

certid_pred =  pd.DataFrame(pred,columns=["pred"])

#print "certid_pred\n", certid_pred

certid_DF = pd.concat([certid_test,certid_pred], axis=1, ignore_index=True)
certid_DF.columns = ["certid","label","pred"]
#print "certid_DF\n",certid_DF
print certid_DF.dtypes

certid_DF.to_csv("certid_DF_drop.csv")


certid_grouped = certid_DF.groupby([certid_DF['certid']])

#certid_grouped = certid_DF.groupby([certid_DF['certid']], as_index=False)

# def label_cnt(arr):  # 同一个人出现次数最多的元素
#     cnt_set = Counter(arr)
#     max_cnt_pair = cnt_set.most_common(1)[0]  # (maxitem,maxcount)
#     return max_cnt_pair[0]

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


true_label_DF = certid_test.drop_duplicates()


compare_df = pd.merge(left=true_label_DF, right=pred_label_DF, how='left', left_on='certid', right_on='certid')


y_test = compare_df["label"]
pred = compare_df["pred-label_cnt"]

cm2=confusion_matrix(y_test,pred)
print  cm2



print "For Person:\n"
result = precision_recall_fscore_support(y_test,pred)
#print result
precision_0 = result[0][0]
recall_0 = result[1][0]
f1_0 = result[2][0]
precision_1 = result[0][1]
recall_1 = result[1][1]
f1_1 = result[2][1]
print "precision_0: ", precision_0,"  recall_0: ", recall_0, "  f1_0: ", f1_0


end_time = datetime.datetime.now()
delta_time = str((end_time-start_time).total_seconds())
print "cost time",delta_time,"s"
