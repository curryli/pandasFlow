# coding=utf-8
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score


# df_1 = pd.read_csv("InnoDeep_1117.csv", sep=',')
#
# df_2 = pd.read_csv("has_risk_mchntcd.csv", sep=',')
#
# df_2 = df_2[(df_2["Bad_mchnt_ratio"]>0.5) & (df_2["Bad_mchntcd_ratio"]>0.5) & (df_2["mchnt_risk_cnt"]>1)]
#
# df_3 = pd.read_csv("has_risk_mchnt.csv", sep=',')
# df_3 = df_3[(df_3["Bad_mchnt_ratio"]>0.5) & (df_3["Bad_mchntcd_ratio"]>0.5) & (df_3["mchnt_risk_cnt"]>1)]


df_1 = pd.read_csv(r"results/InnoDeep_1117.csv", sep=',')
df_1  = df_1[(df_1["label"] == 0)]

df_2 = pd.read_csv(r"results/InnoDeep_1121_noM_low.csv", sep=',')
df_2  = df_2[(df_2["label"] == 0)]
df_3 = pd.read_csv(r"results/InnoDeep_1121_M_high.csv", sep=',')
df_3  = df_3[(df_3["label"] == 0)]


cert_list1 = df_1["certid"].values
cert_list2 = df_2["certid"].values
cert_list3 = df_3["certid"].values

# label_0_list = set(cert_list1).union(set(cert_list2)).union(set(cert_list3))
# print len(label_0_list)
#
# label_0_list = set(cert_list1).union(set(cert_list2).intersection(set(cert_list3)))
# print len(label_0_list)
#
#
#
# label_2_list = set(cert_list2).intersection(set(cert_list3))
# print len(label_2_list)
#
#
# label_2_list = set(cert_list1).intersection(set(cert_list3))
# print len(label_2_list)
# #df_All.to_csv("InnoDeep_1120_nomchnt.csv",index=False)
#
# label_2_list = set(cert_list1).intersection(set(cert_list2))
# print len(label_2_list)
label_2_list = set(cert_list1).intersection(set(cert_list3))
print len(label_2_list)

df_4 = pd.read_csv(r"merged_IF_1123.csv", sep=',')
df_4  = df_4[(df_4["label"] == 0)]
cert_list4 = df_4["certid"].values

label_2_list = set(cert_list1).intersection(set(cert_list4))
print len(label_2_list)


df_5 = pd.read_csv(r"merged.csv", sep=',')
df_5  = df_5[(df_5["label"] == 0)]
cert_list5 = df_5["certid"].values
label_2_list = set(cert_list4).intersection(set(cert_list5))
print len(label_2_list)

df_6 = pd.read_csv(r"merged_split_1123.csv", sep=',')
df_6  = df_6[(df_6["label"] == 0)]
cert_list6 = df_6["certid"].values
label_2_list = set(cert_list5).intersection(set(cert_list6))
print len(label_2_list)


df_5 = pd.read_csv(r"merged.csv", sep=',')
df_6 = pd.read_csv(r"merged_split_1123.csv", sep=',')
df_7 = pd.merge(left=df_6, right=df_5, how='left', left_on='certid', right_on='certid')

def check(x):
    if (x["label_x"]==0) & (x["label_y"]==0):
        return 0
    else:
        return 1


df_7["label"] = df_7.apply(check, axis=1)
df_7 = df_7.fillna(-1)
df_7 = df_7.to_csv("split_1123_5200.csv",index=False)