# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
 

def vote(x):
    cnt_0 = 0
    cnt_1 = 0
    schema = ["a","b","c","d","e","f","g","h","i","j","k","l"]
    #schema = ["a", "b", "c", "d", "e", "f"]
    for item in schema:
        if x[item]==0:
            cnt_0 = cnt_0 + 1
        elif x[item]==1:
            cnt_1 = cnt_1 + 1
    if cnt_0>7:
        return 0
    else:
        return 1

# ############################
df_a = pd.read_csv("test_1.csv", sep=',')
df_b = pd.read_csv("test_2.csv", sep=',')
df_c = pd.read_csv("test_3.csv", sep=',')
df_d = pd.read_csv("test_4.csv", sep=',')
df_e = pd.read_csv("test_5.csv", sep=',')
df_f = pd.read_csv("test_6.csv", sep=',')
df_g = pd.read_csv("test_7.csv", sep=',')
df_h = pd.read_csv("test_8.csv", sep=',')
df_i = pd.read_csv("test_9.csv", sep=',')
df_j = pd.read_csv("test_10.csv", sep=',')
df_k = pd.read_csv("test_11.csv", sep=',')
df_l = pd.read_csv("test_12.csv", sep=',')






df_a = pd.merge(left=df_a, right=df_b, how='left', left_on='certid', right_on='certid')
df_a = pd.merge(left=df_a, right=df_c, how='left', left_on='certid', right_on='certid')
df_a = pd.merge(left=df_a, right=df_d, how='left', left_on='certid', right_on='certid')
df_a = pd.merge(left=df_a, right=df_e, how='left', left_on='certid', right_on='certid')
df_a = pd.merge(left=df_a, right=df_f, how='left', left_on='certid', right_on='certid')
df_a = pd.merge(left=df_a, right=df_g, how='left', left_on='certid', right_on='certid')
df_a = pd.merge(left=df_a, right=df_h, how='left', left_on='certid', right_on='certid')
df_a = pd.merge(left=df_a, right=df_i, how='left', left_on='certid', right_on='certid')
df_a = pd.merge(left=df_a, right=df_j, how='left', left_on='certid', right_on='certid')
df_a = pd.merge(left=df_a, right=df_k, how='left', left_on='certid', right_on='certid')
df_a = pd.merge(left=df_a, right=df_l, how='left', left_on='certid', right_on='certid')



df_a["label"] = df_a.apply(vote, axis=1)

print 17147 - df_a["label"].sum()

df_test = pd.read_csv("test_certid_date_encrypt.csv", sep=',')
df_test = pd.merge(left=df_test, right=df_a, how='left', left_on='certid', right_on='certid')
df_test= df_test.fillna(-1)

df_test.to_csv("merged_split_1123.csv",index=False)