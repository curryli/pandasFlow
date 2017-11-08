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
    train_ori_df = pd.read_csv("small_data.csv", sep=",", low_memory=False, error_bad_lines=False)
    #train_ori_df = pd.read_csv("train_trans_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    #test_ori_df = pd.read_csv("test_small.csv", sep=",", low_memory=False, error_bad_lines=False)
    #test_ori_df = pd.read_csv("test_trans_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    #Trans_ori_df = pd.concat([train_ori_df, test_ori_df], axis=0)

    # Trans_ori_df = Trans_ori_df.fillna(-1)
    #
    # for col in catgory_list:
    #     le = preprocessing.LabelEncoder()
    #     le.fit(Trans_ori_df[col])
    #     Trans_ori_df[col] = le.transform(Trans_ori_df[col])
    #
    # Trans_ori_df = Trans_ori_df.fillna(-1)

    label_df = pd.read_csv("train_label_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)
    Trans_ori_df = pd.merge(left=train_ori_df, right=label_df, how='left', left_on='certid', right_on='certid')


    ###############################
    DateDict = createDateDict((2015, 10, 1), (2017, 5, 1))
    Trans_ori_df["dateNo"] = Trans_ori_df["Settle_dt"].map(lambda x: DateDict[x])
    Trans_ori_df["month"] = Trans_ori_df["Trans_tm"].map(lambda x: getMonth(str(x)))
    Trans_ori_df["date"] = Trans_ori_df["Trans_tm"].map(lambda x: getDate(str(x)))
    Trans_ori_df["hour"] = Trans_ori_df["Trans_tm"].map(lambda x: getHour(str(x)))
    Trans_ori_df["min"] = Trans_ori_df["Trans_tm"].map(lambda x: getMin(str(x)))
    Trans_ori_df["time"] = Trans_ori_df.apply(getTime, axis=1)
    Trans_ori_df["weekday"] = Trans_ori_df["time"].map(lambda x: getWeekday(x))
    Trans_ori_df["stageInMonth"] = Trans_ori_df["date"].map(lambda x: stageInMonth(x))

    Good_df = Trans_ori_df[Trans_ori_df["label"] == 1]
    Bad_df = Trans_ori_df[Trans_ori_df["label"] == 0]


    for col in stat_list:
        new_col = col + "cnt_dis"
        print new_col
        # cnt_dis_col = Trans_ori_df[col].drop_duplicates().shape[0]
        # common_cnt = cnt_dis_col*0.01
        #Trans_ori_df.apply(frequent_item(Trans_ori_df,common_cnt), axis=1)
        # print Good_df[col].value_counts()
        Bad_items = Bad_df[col].value_counts().axes[0].values
        Good_items = Good_df[col].value_counts().axes[0].values
        #risk_items = np.setdiff1d(Bad_items[:int(Bad_items.shape[0]*0.01)], Good_items[:int(Bad_items.shape[0]*0.01)])
        risk_items_1 = np.setdiff1d(Bad_items[:int(Bad_items.shape[0] * 0.1)],Good_items[:int(Bad_items.shape[0] * 0.1)])
        print risk_items_1.size

        risk_items_2 = np.setdiff1d(Bad_items[:int(Bad_items.shape[0] * 0.1)],Good_items[:int(Bad_items.shape[0] * 0.01)])
        print risk_items_2.size

        risk_items_3 = np.setdiff1d(Bad_items[:int(Bad_items.shape[0] * 0.01)],  Good_items[:int(Bad_items.shape[0] * 0.1)])
        print risk_items_3.size

        risk_items_4 = np.setdiff1d(Bad_items[:int(Bad_items.shape[0] * 0.01)],  Good_items[:int(Bad_items.shape[0] * 0.01)])
        print risk_items_4.size
        #print Bad_df[col].value_counts().index



    for col in catgory_less_list:
        new_col = col + "cnt_dis"
        print new_col
        Bad_items = Bad_df[col].value_counts().axes[0].values
        Good_items = Good_df[col].value_counts().axes[0].values

        risk_items = np.setdiff1d(Bad_items,Good_items)
        print Bad_items.size, Good_items.size, risk_items.size


