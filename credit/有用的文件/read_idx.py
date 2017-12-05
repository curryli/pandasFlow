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
from woe_pandas import WOE_pandas
#from month_cnt_func import Month_Cnt_class

catgory_many_list = ['mchnt_cd_filled_idx', 'card_accprt_nm_loc_filled_idx','term_cd_filled_idx',"auth_id_resp_cd_filled_idx"]
catgory_little_list = ['iss_ins_cd_filled_idx', 'trans_chnl_filled_idx', 'mcc_cd_filled_idx', 'resp_cd_filled_idx', 'trans_id_cd_filled_idx', 'orig_trans_st_filled_idx','trans_st_filled_idx', 'trans_curr_cd_filled_idx',
                'fwd_settle_cruu_cd_filled_idx', 'fwd_settle_conv_rt_filled_idx', 'rcv_settle_curr_cd_filled_idx','rcv_settle_conv_rt_filled_idx', 'cdhd_curr_cd_filled_idx',
                'cdhd_conv_rt_filled_idx', 'card_attr_cd_filled_idx','card_media_cd_filled_idx', 'pos_cond_cd_filled_idx', 'pos_entry_md_cd_filled_idx']

catgory_list = catgory_many_list + catgory_little_list


most_frequent_list = catgory_list

#countDistinct_list = most_frequent_list.append("card_no")

math_list = ['RMB']


########################################group 函数########################################
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

def peak_to_peak(arr):
    return arr.max() - arr.min()


#####################################
def cnt_0(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[0]

def cnt_1(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[1]

def cnt_2(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[2]

def cnt_3(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[3]

def cnt_4(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[4]

def cnt_5(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[5]

def cnt_6(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[6]

def cnt_7(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[7]

def cnt_8(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[8]

def cnt_9(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[9]

def cnt_10(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[10]

def cnt_11(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[11]

def cnt_12(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[12]

def cnt_13(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[13]

def cnt_14(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[14]

def cnt_15(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[15]

def cnt_16(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[16]

def cnt_17(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[17]

def cnt_18(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[18]

def cnt_19(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[19]

def cnt_20(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[20]
###################################################


#####################################################################
#####################################################################
###########################非group函数################################
def replace_Not_num(x):
    if ((type(x)==int) | (type(x)==float) | (type(x)==long)):
        return x
    else:
        try:
            result = long(x)
        except ValueError:
            result = -1
        return result

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

def getYear(str):
    if (str[0:4] == "2015"):
        return 0
    elif (str[0:4] == "2016"):
        return 1
    else:
        return 2


def getMonth(str):
    if (len(str) == 9):
        return int(str[0])
    else:
        return int(str[0:2])


def getMonth_No(x):
    return (x["year"]*12 + x["month"]-10)   #最早是2015.10，那么该字段就是0，，2015.10对应1

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


def getWeektime(x):
    return  x["weekday"]*x["hour"]


def stageInMonth(x):
    x_num = int(x)
    if (x_num < 10):
        return 0
    elif (x_num < 20):
        return 1
    else:
        return 2

def is_equal(x,item):
    if(x==item):
        return 1
    else:
        return 0


def getDelta(x,col1,col2):
    return x[col1] - x[col2]

def getRatio(x,col1,col2):
    return float(x[col1])/float(x[col2])

def is_risk_items(x,riskitems):
    if(x in risk_items):
        return int(1)
    else:
        return int(0)


def age_section(x):
    x_num = int(x)
    if (x_num < 22):
        return 0
    elif (x_num < 34):
        return 1
    elif (x_num < 40):
        return 2
    elif (x_num < 55):
        return 3
    else:
        return 4

def has_trans_month(x):  #有几个月有消费
    cnt = 0
    for m in range(0,20):
        colname = r"month_No-cnt_" + str(m)
        if (x[colname]>0):
            cnt = cnt+1
    return cnt

def mean_money_month_list(x):  #有几个月有消费
    mean_list = []
    for m in range(0,20):
        colcnt = r"month_No-cnt_" + str(m)
        colsum = r"month_sum_" + str(m)
        if(x[colcnt]>0):
            mean_list.append(float(x[colsum])/float(x[colcnt]))
        else:
            mean_list.append(0.0)
    return mean_list


#######################################
def month_sum_mean(x,name):  #月消费金额平均
    cnt_list = []
    for m in range(0,20):
        colname = name + str(m)
        cnt_list.append(x[colname])
    return np.mean(cnt_list)

def month_sum_max(x,name):  #月消费金额最大
    cnt_list = []
    for m in range(0,20):
        colname = name + str(m)
        cnt_list.append(x[colname])
    return np.max(cnt_list)

def month_sum_var(x,name):  #月消费金额方差
    cnt_list = []
    for m in range(0,20):
        colname = name + str(m)
        cnt_list.append(x[colname])
    return np.var(cnt_list)

def month_sum_p2p(x,name):  #月消费金额p2p
    cnt_list = []
    for m in range(0,20):
        colname = name + str(m)
        cnt_list.append(x[colname])
    return np.max(cnt_list)- np.min(cnt_list)

#####################################

if __name__ == '__main__':
    reader = pd.read_csv("cert_all_right.csv", low_memory=False, iterator=True)

    loop = True
    chunkSize = 100000
    chunks = []
    i = 0
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
            if (i % 5) == 0:
                print i
            i = i + 1

        except StopIteration:
            loop = False
            print "Iteration is stopped."
    df_All = pd.concat(chunks, ignore_index=True)

    Trans_ori_df = df_All.fillna(-1)

    ###############################

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


##############################################groupby 之后agg################################################
    agg_dict = {}
    for item in most_frequent_list:
        agg_dict[item] = [countDistinct, most_frequent_item, most_frequent_cnt]

    agg_dict["card_no"] = [countDistinct]

    for item in catgory_little_list:
        agg_dict[item] = agg_dict[item] + ['min', 'max', 'mean', 'sum', 'median', 'std', 'var', peak_to_peak]

    is_risk_cols = [("is_risk_"+col) for col in catgory_list]
    for item in is_risk_cols:                 #是不是还要计算高危比例？
        agg_dict[item] = ['sum']

    agg_stat_df = certid_grouped.agg(agg_dict)
    agg_stat_df.columns = agg_stat_df.columns.map('{0[0]}-{0[1]}'.format)


    #print agg_stat_df.columns
    agg_stat_df.to_csv("agg_cat.csv")
#############################################groupby 之后apply##################################################


