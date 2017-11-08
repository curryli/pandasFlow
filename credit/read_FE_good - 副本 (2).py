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


countDistinct_list = ["card_no", "month", "date", "hour", "weekday", "stageInMonth", "Settle_dt", 'iss_ins_cd', 'trans_chnl',
                          'mchnt_cd', 'mcc_cd', 'card_accprt_nm_loc', 'resp_cd', 'trans_id_cd', 'orig_trans_st',
                          'trans_st', 'trans_curr_cd', 'fwd_settle_cruu_cd', 'fwd_settle_conv_rt', 'rcv_settle_curr_cd',
                          'rcv_settle_conv_rt', 'cdhd_curr_cd', 'cdhd_conv_rt', 'term_cd', 'card_attr_cd',
                          'card_media_cd', 'pos_cond_cd', 'pos_entry_md_cd', 'auth_id_resp_cd']


most_frequent_list = ["month", "date", "hour", "weekday", "stageInMonth", 'iss_ins_cd', 'trans_chnl', 'mchnt_cd',
                          'mcc_cd', 'card_accprt_nm_loc', 'resp_cd', 'trans_id_cd', 'orig_trans_st', 'trans_st',
                          'trans_curr_cd', 'fwd_settle_cruu_cd', 'fwd_settle_conv_rt', 'rcv_settle_curr_cd',
                          'rcv_settle_conv_rt', 'cdhd_curr_cd', 'cdhd_conv_rt', 'term_cd', 'card_attr_cd',
                          'card_media_cd', 'pos_cond_cd', 'pos_entry_md_cd', 'auth_id_resp_cd']

math_list = ['Trans_at', 'fwd_settle_at', 'rcv_settle_at', 'cdhd_at', 'dateNo']

catgory_many_list = ['mchnt_cd', 'card_accprt_nm_loc','term_cd',"auth_id_resp_cd"]
catgory_little_list = ['iss_ins_cd', 'trans_chnl', 'mcc_cd', 'resp_cd', 'trans_id_cd', 'orig_trans_st','trans_st', 'trans_curr_cd',
                'fwd_settle_cruu_cd', 'fwd_settle_conv_rt', 'rcv_settle_curr_cd','rcv_settle_conv_rt', 'cdhd_curr_cd',
                'cdhd_conv_rt', 'card_attr_cd','card_media_cd', 'pos_cond_cd', 'pos_entry_md_cd']

catgory_list = catgory_many_list + catgory_little_list

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

def getDelta(x,col1,col2):
    return x[col1] - x[col2]

def getRatio(x,col1,col2):
    return float(x[col1])/float(x[col2])

def is_risk_items(x,riskitems):
    if(x in risk_items):
        return int(1)
    else:
        return int(0)

if __name__ == '__main__':
    #train_ori_df = pd.read_csv("small_data.csv", sep=",", low_memory=False, error_bad_lines=False)
    train_ori_df = pd.read_csv("train_trans_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    #test_ori_df = pd.read_csv("test_small.csv", sep=",", low_memory=False, error_bad_lines=False)
    test_ori_df = pd.read_csv("test_trans_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    Trans_ori_df = pd.concat([train_ori_df, test_ori_df], axis=0)

    label_df = pd.read_csv("train_label_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    Trans_ori_df = pd.merge(left=Trans_ori_df, right=label_df, how='left', left_on='certid', right_on='certid')

    Trans_ori_df = Trans_ori_df.fillna(-1)

    for col in catgory_list:
        le = preprocessing.LabelEncoder()
        le.fit(Trans_ori_df[col])
        Trans_ori_df[col] = le.transform(Trans_ori_df[col])


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

    for col in math_list:
        Trans_ori_df[col] = Trans_ori_df[col].apply(replace_Not_num)

    Good_df = Trans_ori_df[Trans_ori_df["label"] == 1]
    Bad_df = Trans_ori_df[Trans_ori_df["label"] == 0]



    for col in catgory_many_list:
        Bad_items = Bad_df[col].value_counts().axes[0].values
        Good_items = Good_df[col].value_counts().axes[0].values
        risk_items = np.setdiff1d(Bad_items[:int(Bad_items.shape[0] * 0.01)], Good_items[:int(Bad_items.shape[0] * 0.1)])
        new_col = "is_risk_" + col
        Trans_ori_df[new_col] = Trans_ori_df[col].map(lambda x: is_risk_items(x,risk_items))

    for col in catgory_little_list:
        Bad_items = Bad_df[col].value_counts().axes[0].values
        Good_items = Good_df[col].value_counts().axes[0].values
        risk_items = np.setdiff1d(Bad_items,Good_items)
        new_col = "is_risk_" + col
        Trans_ori_df[new_col] = Trans_ori_df[col].map(lambda x: is_risk_items(x,risk_items))

    Trans_ori_df = Trans_ori_df.fillna(-1)



####################################################################################################
    Trans_df = Trans_ori_df

    #print Trans_df.dtypes

    certid_grouped = Trans_df.groupby([Trans_df['certid']], group_keys=True)


    group_keys = []
    for name, group in certid_grouped:
        group_keys.append(name)

    agg_dict = {}

    for item in countDistinct_list:
        agg_dict[item] = [countDistinct]

    for item in most_frequent_list:
        agg_dict[item] = [most_frequent_item, most_frequent_cnt]

    for item in math_list:
        agg_dict[item] = ['min', 'max', 'mean', 'sum', 'median', 'std', 'var', peak_to_peak]

    agg_dict['Trans_at'] = ['count', 'min', 'max', 'mean', 'sum', 'median', 'std', 'var', peak_to_peak]

    is_risk_cols = [("is_risk_"+col) for col in catgory_list]
    for item in is_risk_cols:                 #是不是还要计算高危比例？
        agg_dict[item] = ['sum']

     

    stat_df = certid_grouped.agg(agg_dict)
    stat_df.columns = stat_df.columns.map('{0[0]}-{0[1]}'.format)


    stat_df["tras_at_max_mean_ratio"] = stat_df.apply(lambda df:  getRatio(df, "Trans_at-max", "Trans_at-mean"), axis=1)

    #print stat_df.columns
    stat_df.to_csv("temp.csv")

    ########################################################################################################
    stat_df = pd.read_csv("temp.csv", sep=",", low_memory=False, error_bad_lines=False)

    train_certid_df = pd.read_csv("train_certid_date_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)
    test_certid_df = pd.read_csv("test_certid_date_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    certid_df = pd.concat([train_certid_df, test_certid_df], axis=0)
    certid_df["apply_dateNo"] = certid_df["first_time"].map(lambda x: DateDict[x])

    certid_dummies = pd.get_dummies(certid_df[["sex", "aera_code"]])

    certid_dummies = pd.concat([certid_df[['certid',"apply_dateNo"]], certid_dummies], axis=1)

    age_df = certid_df[['certid', 'age']]  # .to_frame()

    Effect_df = pd.merge(left=certid_dummies, right=age_df, how='left', left_on='certid', right_on='certid')

    Effect_df = pd.merge(left=Effect_df, right=stat_df, how='left', left_on='certid', right_on='certid')

    Effect_df["apply_max_delta"] = Effect_df.apply(lambda df:  getDelta(df, "dateNo-max", "apply_dateNo"), axis=1)
    Effect_df["apply_mean_delta"] = Effect_df.apply(lambda df: getDelta(df, "dateNo-mean", "apply_dateNo"), axis=1)
    Effect_df["min_apply_delta"] = Effect_df.apply(lambda df: getDelta(df, "apply_dateNo", "dateNo-min"), axis=1)

    Effect_df = pd.merge(left=Effect_df, right=label_df, how='left', left_on='certid', right_on='certid')

    Effect_df.to_csv("train.csv",index=False)