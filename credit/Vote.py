# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
 


df_All = pd.read_csv("1120_mchnt.csv", sep=',')


def vote(x):
    cnt_0 = 0
    cnt_1 = 0
    schema = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u"]
    #schema = ["a", "b", "c", "d", "e", "f"]
    for item in schema:
        if x[item]==0:
            cnt_0 = cnt_0 + 1
        elif x[item]==1:
            cnt_1 = cnt_1 + 1
    if cnt_0>6:
        return 0
    else:
        return 1


df_All["result"] = df_All.apply(vote, axis=1)

df_All.to_csv("new_1120_mchnt.csv",index=False)

# ############################
# df_test = pd.read_csv("test_certid_date_encrypt.csv", sep=',')
# certid_test_DF = df_test[["certid"]]
#
#
# df_All = pd.merge(left=certid_test_DF, right=df_All, how='left', left_on='certid', right_on='certid')
# df_All.fillna(-1)
#
# df_All.to_csv("InnoDeep_1121_M_high.csv",index=False)