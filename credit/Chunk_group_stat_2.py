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

label_list = ["money_near_last","1hour_highrisk_MCC_cnt","min15_highrisk_MCC_cnt","day30_highrisk_MCC_cnt","is_Mchnt_changed","day7_freq_cnt","1hour_failure_cnt","2hour_failure_cnt","day7_no_trans","cur_success_cnt","money_eq_last","hist_highrisk_MCC_cnt","hist_query_cnt","day3_tot_cnt","day30_no_trans","min15_failure_cnt","1hour_no_trans","is_bigRMB_1000","cur_failure_cnt","is_highrisk_MCC","cur_highrisk_MCC_cnt","cardholder_fail","is_bigRMB_500","is_PW_need","is_lowrisk_MCC","is_mcc_changed","RMB_bits","is_spec_airc","is_success","cur_tot_cnt","is_norm_rate","hist_freq_cnt","no_auth_id_resp_cd","hist_no_trans","is_large_integer","count_89","is_weekend","is_Night"]
################################################# 

reader = pd.read_csv("cert_all_right.csv", low_memory=False, iterator=True)
#reader = pd.read_csv("trans_small.csv", low_memory=False, iterator=True)


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
#print df_All.columns

df_All = shuffle(df_All)


def bit_more(x):
    if(x>3):
        return 1
    else:
        return 0

df_All["bit_more"] = df_All["RMB_bits"].map(lambda x: bit_more(x))


def count89_more(x):
    if(x>3):
        return 1
    else:
        return 0

df_All["count89_more"] = df_All["count_89"].map(lambda x: count89_more(x))


def frequent_trans(x):
    if((x>=0) & (x<3)):
        return 1
    else:
        return 0

df_All["frequent_trans"] = df_All["quant_interval_1"].map(lambda x: frequent_trans(x))



certid_grouped = df_All.groupby([df_All['certid']])

def cnt_0(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[0]

def cnt_1(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[1]

label_list.append("bit_more")
label_list.append("count89_more")
label_list.append("frequent_trans")

agg_dict = {}
for item in label_list:
    agg_dict[item] = ['sum','mean',"std", cnt_0, cnt_1]

agg_stat_df = certid_grouped.agg(agg_dict)


agg_stat_df.columns = agg_stat_df.columns.map('{0[0]}-{0[1]}'.format)

agg_stat_df.reset_index(level=0, inplace=True)
#
# def count0(x):
#     return  x["weekday"]*x["hour"]
#
# agg_stat_df = agg_stat_df.apply(count0, axis=1)

agg_stat_df.to_csv("translabel_stat_2.csv",index=False)

