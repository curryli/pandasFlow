# coding=utf-8
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score


df_1 = pd.read_csv("InnoDeep_1117.csv", sep=',')

df_2 = pd.read_csv("has_risk_mchntcd.csv", sep=',')

df_3 = pd.read_csv("has_risk_mchnt.csv", sep=',')

df_1  = df_1[(df_1["label"] == 0)]

print df_1[["certid"]].values
print df_2[["certid"]].values
print df_3[["certid"]].values
#df_All.to_csv("InnoDeep_1120_nomchnt.csv",index=False)


