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

start_time = datetime.datetime.now()


################################################# 



certid_DF = pd.read_csv("certid_DF_drop.csv", sep=",", low_memory=False, error_bad_lines=False)


certid_grouped = certid_DF.groupby([certid_DF['certid']])



def label_cnt(arr):  # 同一个人出现次数最多的元素
    cnt_0 = 0
    cnt_1 = 1
    arr_values = arr.values
    for i in range(len(arr_values)):
        if arr_values[i]==float(0):
            cnt_0 = cnt_0+1
        elif arr_values[i]==float(1):
            cnt_1 = cnt_1+1
    if(cnt_0>0):
        return 0
    elif (cnt_1<2):
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

true_label_DF = certid_DF[["certid","label"]].drop_duplicates()


compare_df = pd.merge(left=true_label_DF, right=pred_label_DF, how='left', left_on='certid', right_on='certid')


y_test = compare_df["label"]
pred = compare_df["pred-label_cnt"]

cm2=confusion_matrix(y_test,pred)
print  cm2



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



