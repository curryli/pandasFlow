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
from collections import Counter
import time, datetime
from sklearn import preprocessing
from dateutil import parser
import woe_pandas
from woe_pandas import WOE_pandas

catgory_many_list = ['mchnt_cd', 'card_accprt_nm_loc','term_cd',"auth_id_resp_cd"]
catgory_little_list = ['iss_ins_cd', 'trans_chnl', 'mcc_cd', 'resp_cd', 'trans_id_cd', 'orig_trans_st','trans_st', 'trans_curr_cd',
                'fwd_settle_cruu_cd', 'fwd_settle_conv_rt', 'rcv_settle_curr_cd','rcv_settle_conv_rt', 'cdhd_curr_cd',
                'cdhd_conv_rt', 'card_attr_cd','card_media_cd', 'pos_cond_cd', 'pos_entry_md_cd']

catgory_list = catgory_many_list + catgory_little_list

cal_catgory_list = ["month", "date", "hour", "weekday", "stageInMonth"]

most_frequent_list = catgory_list + cal_catgory_list


math_list = ['Trans_at', 'fwd_settle_at', 'rcv_settle_at', 'dateNo']  #'cdhd_at',


if __name__ == '__main__':
    df_All = pd.read_csv("train.csv", sep=',')
    #df_All = df_All[(df_All["label"] == 0) | (df_All["label"] == 1)]
    #print df_All
    df_All = df_All.fillna(-1)


    Effect_df = df_All

    #print Effect_df
    Effect_df_dup = Effect_df


    def AddWOE_nominal(df, VarName, VarWOE):
        dict1 = dict.fromkeys(VarWOE['class'])
        j = 0
        for key in dict1:
            dict1[key] = VarWOE['woe'][j]
            j = j + 1
        new_col = VarName + "_woe"
        df[new_col] = df[VarName].map(lambda x: dict1[x])
        return df


    ########################################Add WOE#############################################
    wp1 = WOE_pandas(Effect_df)
    for item in most_frequent_list:
        for func in ["countDistinct", "most_frequent_item", "most_frequent_cnt"]:
            col = item + "-" + func
            WOE_nominal = wp1.CalcWOE_nominal(col, 'label')
            Effect_df = wp1.AddWOE_nominal(col, WOE_nominal)


    for item in math_list:
        for func in ['min', 'max', 'mean', 'sum', 'median', 'std', 'var', "peak_to_peak"]:
            col = item + "-" + func
            WOE_bin = wp1.CalcWOE_bin(col, 'label', 10)
            Effect_df = wp1.AddWOE_bin(col, WOE_bin)


    Effect_df.to_csv("train_add_woe.csv", index=False)
    ###########################################################################################

    # ########################################Replace WOE#############################################
    # wp2 = WOE_pandas(Effect_df_dup)
    # for item in most_frequent_list:
    #     for func in ["countDistinct", "most_frequent_item", "most_frequent_cnt"]:
    #         col = item + "-" + func
    #         WOE_nominal = wp2.CalcWOE_nominal(col, 'label')
    #         wp2.ReplaceWOE_nominal(col, WOE_nominal)
    #
    # for item in math_list:
    #     for func in ['min', 'max', 'mean', 'sum', 'median', 'std', 'var', "peak_to_peak"]:
    #         col = item + "-" + func
    #         WOE_bin = wp2.CalcWOE_bin(col, 'label', 10)
    #         Effect_df_dup = wp2.ReplaceWOE_bin(col, WOE_bin)
    #
    # Effect_df_dup.to_csv("train_replace_woe.csv", index=False)
    # ###########################################################################################



    # print Effect_df.dtypes.to_csv("temp2.csv",index=True)





