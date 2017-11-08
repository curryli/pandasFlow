# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
# from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import os  # python miscellaneous OS system tool
from collections import Counter
import time, datetime
from sklearn import preprocessing

catgory_list = ['iss_ins_cd', 'trans_chnl', 'mchnt_cd', 'mcc_cd', 'card_accprt_nm_loc', 'resp_cd', 'trans_id_cd', 'orig_trans_st',
                          'trans_st', 'trans_curr_cd', 'fwd_settle_cruu_cd', 'fwd_settle_conv_rt', 'rcv_settle_curr_cd',
                          'rcv_settle_conv_rt', 'cdhd_curr_cd', 'cdhd_conv_rt', 'term_cd', 'card_attr_cd',
                          'card_media_cd', 'pos_cond_cd', 'pos_entry_md_cd', 'auth_id_resp_cd']

countDistinct_list = ["month", "date", "hour", "weekday", "stageInMonth", "Settle_dt", 'iss_ins_cd', 'trans_chnl',
                          'mchnt_cd', 'mcc_cd', 'card_accprt_nm_loc', 'resp_cd', 'trans_id_cd', 'orig_trans_st',
                          'trans_st', 'trans_curr_cd', 'fwd_settle_cruu_cd', 'fwd_settle_conv_rt', 'rcv_settle_curr_cd',
                          'rcv_settle_conv_rt', 'cdhd_curr_cd', 'cdhd_conv_rt', 'term_cd', 'card_attr_cd',
                          'card_media_cd', 'pos_cond_cd', 'pos_entry_md_cd', 'auth_id_resp_cd']


most_frequent_list = ["month", "date", "hour", "weekday", "stageInMonth", 'iss_ins_cd', 'trans_chnl', 'mchnt_cd',
                          'mcc_cd', 'card_accprt_nm_loc', 'resp_cd', 'trans_id_cd', 'orig_trans_st', 'trans_st',
                          'trans_curr_cd', 'fwd_settle_cruu_cd', 'fwd_settle_conv_rt', 'rcv_settle_curr_cd',
                          'rcv_settle_conv_rt', 'cdhd_curr_cd', 'cdhd_conv_rt', 'term_cd', 'card_attr_cd',
                          'card_media_cd', 'pos_cond_cd', 'pos_entry_md_cd', 'auth_id_resp_cd']

#math_list = ['Trans_at', 'fwd_settle_at', 'rcv_settle_at', 'cdhd_at', 'dateNo']
math_list = ['Trans_at', 'fwd_settle_at', 'rcv_settle_at', 'dateNo']
#math_list = ['Trans_at']


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
        return str[0]
    else:
        return str[0:2]


def getDate(str):
    return str[-8:-6]


def getHour(str):
    return str[-6:-4]


def getMin(str):
    return str[-4:-2]


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


# def getAverage(arr):
#     arr_np = np.array(arr)
#     return arr_np.mean()
#



if __name__ == '__main__':

    from dateutil import parser

    Trans_df = pd.read_csv("train_trans_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    Trans_df['Trans_at'].to_csv("Trans_at.csv")

    # print trans_df.trans_id_cd.to_csv("save.txt")


    certid_grouped = Trans_df.groupby([Trans_df['certid']], group_keys=True)

    # print certid_grouped.keys


    group_keys = []
    for name, group in certid_grouped:
        group_keys.append(name)
        # print group_keys

##############################

    # for col in catgory_list:
    #     le = preprocessing.LabelEncoder()
    #     le.fit(Trans_ori_df[col])
    #     Trans_ori_df[col] = le.transform(Trans_ori_df[col])
    #
    # #print Trans_ori_df
    # Trans_df =Trans_ori_df

###############################
    # DateDict = createDateDict((2015, 10, 1), (2017, 5, 1))
    # Trans_df["dateNo"] = Trans_df["Settle_dt"].map(lambda x: DateDict[x])
    #
    # Trans_df["month"] = Trans_df["Trans_tm"].map(lambda x: getMonth(str(x)))
    # Trans_df["date"] = Trans_df["Trans_tm"].map(lambda x: getDate(str(x)))
    # Trans_df["hour"] = Trans_df["Trans_tm"].map(lambda x: getHour(str(x)))
    # Trans_df["min"] = Trans_df["Trans_tm"].map(lambda x: getMin(str(x)))
    #
    # Trans_df["time"] = Trans_df.apply(getTime, axis=1)
    # Trans_df["weekday"] = Trans_df["time"].map(lambda x: getWeekday(x))
    # Trans_df["stageInMonth"] = Trans_df["date"].map(lambda x: stageInMonth(x))

   # Trans_df.to_csv("temp_1.csv")

    agg_dict = {}

    # for item in countDistinct_list:
    #     agg_dict[item] = [countDistinct]
    #
    # for item in most_frequent_list:
    #     agg_dict[item] = [most_frequent_item, most_frequent_cnt]

    # for item in math_list:
    #     agg_dict[item] = ['min', 'max', 'mean', 'sum', 'median', 'std', 'var', peak_to_peak]


    agg_dict = {'Trans_at': ['min', 'max', 'mean']}


    stat_df = certid_grouped.agg(agg_dict)
    stat_df.columns = stat_df.columns.map('{0[0]}-{0[1]}'.format)
    stat_df.to_csv("temp.csv")




    ########################################################################################################
    stat_df = pd.read_csv("temp.csv", sep=",", low_memory=False, error_bad_lines=False)

    certid_df = pd.read_csv("train_certid_date_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    certid_dummies = pd.get_dummies(certid_df[["sex", "aera_code"]])

    certid_dummies = pd.concat([certid_df[['certid']], certid_dummies], axis=1)

    age_df = certid_df[['certid', 'age']]  # .to_frame()

    Effect_df = pd.merge(left=age_df, right=certid_dummies, how='left', left_on='certid', right_on='certid')

    label_df = pd.read_csv("train_label_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    Effect_df = pd.merge(left=Effect_df, right=stat_df, how='left', left_on='certid', right_on='certid')

    Effect_df = pd.merge(left=Effect_df, right=label_df, how='left', left_on='certid', right_on='certid')

    Effect_df.to_csv("train.csv")