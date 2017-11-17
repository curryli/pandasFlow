# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
 


df_All = pd.read_csv("xgboost_results_1117.csv", sep=',')


def vote(x):
    cnt_0 = 0
    cnt_1 = 0
    schema = ["a","b","c","d","e","f","g","h","i","j"]
    for item in schema:
        if x[item]==0:
            cnt_0 = cnt_0 + 1
        elif x[item]==1:
            cnt_1 = cnt_1 + 1
    if cnt_0>cnt_1:
        return 0
    else:
        return 1


df_All["result"] = df_All.apply(vote, axis=1)

df_All.to_csv("result_1117_vote.csv",index=False)
