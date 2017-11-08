# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import os                          #python miscellaneous OS system tool


#Trans_df = pd.read_csv("small_data.csv",sep=",", low_memory=False, error_bad_lines=False)
Trans_df = pd.read_csv("train_trans_encrypt.csv",sep=",", low_memory=False, error_bad_lines=False)

Trans_df['Trans_at'].to_csv("Trans_at.csv")

#print trans_df.trans_id_cd.to_csv("save.txt")


certid_grouped = Trans_df.groupby([Trans_df['certid']], group_keys=True)


#print certid_grouped.keys


group_keys = []
for name, group in certid_grouped:
    group_keys.append(name)
#print group_keys

agg_dict = {'Trans_at':['min','max','mean']}
agg_dict ={}

item_list = ['iss_ins_cd','trans_chnl','mchnt_cd','mcc_cd','card_accprt_nm_loc','resp_cd','trans_id_cd','orig_trans_st','trans_st','trans_curr_cd','fwd_settle_cruu_cd','fwd_settle_conv_rt','rcv_settle_curr_cd','rcv_settle_conv_rt','cdhd_curr_cd','cdhd_conv_rt','term_cd','card_attr_cd','card_media_cd','retri_ref_no','pos_cond_cd','pos_entry_md_cd','auth_id_resp_cd']


def countditinct(arr):
    arr_set = set(arr)
    return len(arr_set)


#item_list = ['trans_chnl']
for item in item_list:
    agg_dict[item] = ['count']



#stat_df = certid_grouped.agg({'Trans_at':['min','max','mean'],'trans_chnl':['count']})

stat_df = certid_grouped.agg(agg_dict)

#print stat_df.xs('Trans_at',axis=1).xs('min', axis=1)

#stat_df = certid_grouped.agg({'Trans_at':['min','max']})

# print stat_df.columns
#stat_df.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in stat_df.columns]
#stat_df.columns = stat_df.columns.map('|'.join)
stat_df.columns = stat_df.columns.map('{0[0]}-{0[1]}'.format)
#print stat_df.columns
#print stat_df
#print stat_df.iloc[:, stat_df.columns.get_level_values(1)=='min']

stat_df.to_csv("temp.csv")
stat_df = pd.read_csv("temp.csv",sep=",", low_memory=False, error_bad_lines=False)
#stat_df = pd.concat([pd.DataFrame(group_keys, columns=['certid'])['certid']  ,stat_df], axis=1)

#print stat_df

# result2= certid_grouped.trans_chnl.nunique()
# result3= certid_grouped.trans_chnl.apply(lambda x: len(x.unique()))
# #result2= Trans_df.groupby([Trans_df['certid'],Trans_df['trans_chnl']]).apply(lambda x: len(x.unique()))
# print result2,result3



certid_df = pd.read_csv("train_certid_date_encrypt.csv",sep=",", low_memory=False, error_bad_lines=False)

certid_dummies = pd.get_dummies(certid_df[["sex","aera_code"]])

certid_dummies = pd.concat([certid_df[['certid']],certid_dummies], axis=1)
#print certid_dummies.columns


age_df = certid_df[['certid','age']]  #.to_frame()
#print age_df

Effect_df = pd.merge(left=age_df, right=certid_dummies, how='left', left_on='certid', right_on='certid')
#print Effect_df
label_df = pd.read_csv("train_label_encrypt.csv",sep=",", low_memory=False, error_bad_lines=False)

Effect_df = pd.merge(left=Effect_df, right=stat_df, how='left', left_on='certid', right_on='certid')
#print Effect_df

Effect_df = pd.merge(left=Effect_df, right=label_df, how='left', left_on='certid', right_on='certid')

Effect_df.to_csv("result.csv")