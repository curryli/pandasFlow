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

catgory_less_list = ['iss_ins_cd', 'trans_chnl', 'mcc_cd', 'resp_cd', 'trans_id_cd', 'orig_trans_st','trans_st', 'trans_curr_cd',
                'fwd_settle_cruu_cd', 'fwd_settle_conv_rt', 'rcv_settle_curr_cd','rcv_settle_conv_rt', 'cdhd_curr_cd',
                'cdhd_conv_rt', 'card_attr_cd','card_media_cd', 'pos_cond_cd', 'pos_entry_md_cd']

stat_list = ['mchnt_cd', 'card_accprt_nm_loc','term_cd',"auth_id_resp_cd"]

def countDistinct(arr):
    arr_set = set(arr)
    return len(arr_set)


def most_frequent_item(arr):  # 同一个人出现次数最多的元素
    cnt_set = Counter(arr)
    max_cnt_pair = cnt_set.most_common(1)[0]  # (maxitem,maxcount)
    return max_cnt_pair[0]


def most_frequent_cnt(arr):
    cnt_set = Counter(arr)
    max_cnt_pair = cnt_set.most_common(1)[0]  # (maxitem,maxcount)
    return max_cnt_pair[1]

def replace_Not_num(x):
    if ((type(x)==int) | (type(x)==float) | (type(x)==long)):
        return x
    else:
        return -1

def createDateDict(start, end):
    start_date = datetime.date(*start)
    end_date = datetime.date(*end)

    datedict = {}
    idx = 0
    curr_date = start_date
    while curr_date != end_date:
        date_str = "%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day)
        datedict[long(date_str)] = idx
        curr_date += datetime.timedelta(1)
        idx = idx + 1

    return datedict


def peak_to_peak(arr):
    return arr.max() - arr.min()


def getMonth(str):
    if (len(str) == 9):
        return int(str[0])
    else:
        return int(str[0:2])


def getDate(str):
    return int(str[-8:-6])


def getHour(str):
    return int(str[-6:-4])


def getMin(str):
    return int(str[-4:-2])


def getTime(x):
    return str(x["Settle_dt"]) + str(x["Trans_tm"])[-6:]


def getWeekday(x):
    return parser.parse(x).weekday()


def stageInMonth(x):
    x_num = int(x)
    if (x_num < 10):
        return 0
    elif (x_num < 20):
        return 1
    else:
        return 2

def frequent_item(arr,return_num):  #返回前10%
    cnt_set = Counter(arr)
    frequent_cnt_pair = cnt_set.most_common(return_num)[0]  # (maxitem,maxcount)
    return frequent_cnt_pair[0]

if __name__ == '__main__':
    #train_ori_df = pd.read_csv("small_data.csv", sep=",", low_memory=False, error_bad_lines=False)
    train_ori_df = pd.read_csv("train_trans_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    label_df = pd.read_csv("train_label_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)
    Trans_ori_df = pd.merge(left=train_ori_df, right=label_df, how='left', left_on='certid', right_on='certid')



    Trans_ori_df.loc[0:2, ['label']] = 1
    Trans_ori_df.loc[3:5, ['label']] = -1


    def cnt_pair(gs):
        cnt_keys = gs.axes[0].values
        cnt_values = gs.values
        cnt_dict = {}
        for i in range(0, len(cnt_keys)):
            cnt_dict[cnt_keys[i]] = cnt_values[i]

        if cnt_dict.has_key(0):
            bad_cnt = cnt_dict[0]
        else:
            bad_cnt =0

        if cnt_dict.has_key(1):
            good_cnt = cnt_dict[1]
        else:
            good_cnt =0

        return (bad_cnt,good_cnt)


    # print Trans_ori_df
    Grp_all = Trans_ori_df["label"].value_counts()
    total_bad = cnt_pair(Grp_all)[0]
    total_good = cnt_pair(Grp_all)[1]


    print total_good

    def CalcWOE_nominal(df, VarName, label):  #名义变量WOE
        WOE_Map = pd.DataFrame()
        Vars = np.unique(df[VarName])  #该列的取值distinct数组
        for v in Vars:
            tmp = df[VarName] == v
            tmp_df = df[tmp]
            grp = tmp_df["label"].value_counts()
            Bad = cnt_pair(grp)[0]
            Good = cnt_pair(grp)[1]
            good_ratio = float(Good) / total_good
            bad_ratio = float(Bad) / total_bad
            WOE = np.log(bad_ratio / good_ratio)
            IV = (bad_ratio - good_ratio) * WOE
            result = pd.DataFrame([[VarName, v, WOE, IV]], index=None, columns=['variable', 'class', 'woe', 'iv'])
            WOE_Map = WOE_Map.append(result, ignore_index=True)
        return WOE_Map


    print "CalcWOE_nominal"
    print CalcWOE_nominal(Trans_ori_df, 'trans_chnl', 'label')


    def CalcWOE_bin(df, VarName, label, N):   #N是对数值型变量N等分
        WOE_Map = pd.DataFrame()
        max_value = max(df[VarName])
        min_value = min(df[VarName])
        bin = float(max_value - min_value) / N
        for i in range(N):
            bin_U = min_value + (i + 1) * bin
            bin_L = bin_U - bin
            if i == 1:
                tmp = (df[VarName] >= bin_L) & (df[VarName] <= bin_U)
                tmp_df = df[tmp]
                grp = tmp_df["label"].value_counts()
            else:
                tmp = (df[VarName] > bin_L) & (df[VarName] <= bin_U)
                tmp_df = df[tmp]
                grp = tmp_df["label"].value_counts()
            Bad = cnt_pair(grp)[0]
            Good = cnt_pair(grp)[1]
            good_ratio = float(Good) / total_good
            bad_ratio = float(Bad) / total_bad
            WOE = np.log(bad_ratio / good_ratio)
            IV = (bad_ratio - good_ratio) * WOE
            result = pd.DataFrame([[VarName, [bin_L, bin_U, WOE], WOE, IV]],
                                  index=None, columns=['variable', 'class+woe', 'woe', 'iv'])
            WOE_Map = WOE_Map.append(result, ignore_index=True)
        return WOE_Map


    print "CalcWOE_bin"
    print CalcWOE_bin(Trans_ori_df, 'Trans_at', 'label', 5)


    def ReplaceWOE_nominal(VarName, SourceDF, VarWOE):
        dict1 = dict.fromkeys(VarWOE['class'])
        j = 0
        for key in dict1:
            dict1[key] = VarWOE['woe'][j]
            j = j + 1
        SourceDF[VarName] = SourceDF[VarName].map(dict1)
        return SourceDF


    def ReplaceWOE_bin(VarName, SourceDF, VarWOE):
        items = np.unique(SourceDF[VarName])
        m = min(SourceDF[VarName])
        dict2 = {}
        for it in items:
            if it == m:
                dict2[it] = VarWOE['class+woe'][0][2]
            else:
                for l, u, w in VarWOE['class+woe']:
                    if (it > l) & (it <= u):
                        dict2[it] = w
        SourceDF[VarName] = SourceDF[VarName].map(dict2)
        return SourceDF



