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



cal_catgory_list = ["dateNo", "month", "date", "hour", "weekday", "stageInMonth"]

most_frequent_list = cal_catgory_list


math_list = ['Trans_at', 'fwd_settle_at', 'rcv_settle_at', 'cdhd_at']


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
    result = -1
    try:
        result = parser.parse(x).weekday()
    except ValueError:
        result = -1
    return result


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
    if(float(x[col2])==0.0):
        return -1
    else:
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

def month_sum_var(x,name):  #月消费金额或者次数方差
    cnt_list = []
    for m in range(0,20):
        colname = name + str(m)
        cnt_list.append(x[colname])
    return np.var(cnt_list)

def month_sum_p2p(x,name):  #月消费金额或者次数p2p
    cnt_list = []
    for m in range(0,20):
        colname = name + str(m)
        cnt_list.append(x[colname])
    return np.max(cnt_list)- np.min(cnt_list)

def month_sum_stable(x,name):  #月消费金额或者次数的稳定性
    cnt_list = []
    for m in range(0,20):
        colname = name + str(m)
        cnt_list.append(x[colname])
    avg = np.mean(cnt_list)
    if (avg>0):
        avg_list = []
        for m in range(0,20):
            colname = name + str(m)
            avg_list.append(float(x[colname])/float(avg))
        return np.var(avg_list)
    else:
        return -1

#####################################

if __name__ == '__main__':
    #train_ori_df = pd.read_csv("small_data.csv", sep=",", low_memory=False, error_bad_lines=False)
    train_ori_df = pd.read_csv("train_spark.csv", sep=",", low_memory=False, error_bad_lines=False)
    #print train_ori_df.shape

    #test_ori_df = pd.read_csv("test_small.csv", sep=",", low_memory=False, error_bad_lines=False)
    test_ori_df = pd.read_csv("test_spark.csv", sep=",", low_memory=False, error_bad_lines=False)
    #print  test_ori_df.shape

    Trans_ori_df = pd.concat([train_ori_df, test_ori_df], axis=0)
    #print  Trans_ori_df.shape

    #Trans_ori_df = Trans_ori_df[Trans_ori_df["card_attr_cd"]!="01"]
    #print Trans_ori_df.shape


    label_df = pd.read_csv("train_label_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    Trans_ori_df = pd.merge(left=Trans_ori_df, right=label_df, how='left', left_on='certid', right_on='certid')

    Trans_ori_df = Trans_ori_df.fillna(-1)
    #Trans_ori_df = Trans_ori_df[Trans_ori_df["certid"] != -1 & Trans_ori_df["Settle_dt"] != -1.0]
    Trans_ori_df[["trans_id_cd"]].to_csv("trans_id_cd.csv", index=False)

    Trans_ori_df["fund_shortage"] = Trans_ori_df["resp_cd"].map(lambda x: is_equal(x,"YD51"))  #是否出现资金不足


    ###############################
    DateDict = createDateDict((2015, 10, 1), (2017, 5, 1))
    Trans_ori_df["dateNo"] = Trans_ori_df["Settle_dt"].map(lambda x: DateDict[x])
    Trans_ori_df["year"] = Trans_ori_df["Settle_dt"].map(lambda x: getYear(str(x)))
    Trans_ori_df["month"] = Trans_ori_df["Trans_tm"].map(lambda x: getMonth(str(x)))
    Trans_ori_df["date"] = Trans_ori_df["Trans_tm"].map(lambda x: getDate(str(x)))
    Trans_ori_df["hour"] = Trans_ori_df["Trans_tm"].map(lambda x: getHour(str(x)))
    Trans_ori_df["min"] = Trans_ori_df["Trans_tm"].map(lambda x: getMin(str(x)))
    Trans_ori_df["time"] = Trans_ori_df.apply(getTime, axis=1)
    Trans_ori_df["weekday"] = Trans_ori_df["time"].map(lambda x: getWeekday(x))
    Trans_ori_df["weektime"] = Trans_ori_df.apply(getWeektime, axis=1)
    Trans_ori_df["stageInMonth"] = Trans_ori_df["date"].map(lambda x: stageInMonth(x))

    Trans_ori_df["month_No"] = Trans_ori_df.apply(getMonth_No, axis=1)


###########################################################################################################################################################################
    for col in math_list:
        Trans_ori_df[col] = Trans_ori_df[col].apply(replace_Not_num)

    Good_df = Trans_ori_df[Trans_ori_df["label"] == 1]
    Bad_df = Trans_ori_df[Trans_ori_df["label"] == 0]

    Trans_ori_df = Trans_ori_df.fillna(-1)

####################################################################################################
    Trans_df = Trans_ori_df

    certid_grouped = Trans_df.groupby([Trans_df['certid']], group_keys=True)


    group_keys = []
    for name, group in certid_grouped:
        group_keys.append(name)


##############################################groupby 之后agg################################################
    agg_dict = {}
    for item in most_frequent_list:
        agg_dict[item] = [countDistinct, most_frequent_item, most_frequent_cnt]

    for item in math_list:
        agg_dict[item] = ['min', 'max', 'mean', 'sum', 'median', 'std', 'var', peak_to_peak]

    agg_dict["card_no"] = [countDistinct]
    agg_dict['Trans_at'].append('count')

    for item in cal_catgory_list:
        agg_dict[item] = agg_dict[item] + ['min', 'max', 'mean', 'sum', 'median', 'std', 'var', peak_to_peak]

    agg_dict["month_No"] = [ cnt_0, cnt_1, cnt_2, cnt_3, cnt_4, cnt_5, cnt_6, cnt_7, cnt_8, cnt_9, cnt_10, cnt_11, cnt_12, cnt_13, cnt_14, cnt_15, cnt_16, cnt_17, cnt_18, cnt_19, cnt_20]

    agg_dict["fund_shortage"] = ['sum']

    agg_stat_df = certid_grouped.agg(agg_dict)
    agg_stat_df.columns = agg_stat_df.columns.map('{0[0]}-{0[1]}'.format)

    agg_stat_df["tras_at_max_mean_ratio"] = agg_stat_df.apply(lambda df:  getRatio(df, "Trans_at-max", "Trans_at-mean"), axis=1)

    #agg_stat_df = agg_stat_df.fillna(-1)

    #print agg_stat_df.columns
    agg_stat_df.to_csv("agg_stat_temp.csv")
#############################################groupby 之后apply##################################################

    # def month_sum_m0(subdf): #可以把每一个group过后的东西都是一个子dataframe，函数里面完全按照dataframe操作即可
    #     return subdf[subdf["month_No"]==0]["Trans_at"].sum()

    ####################金额##############################
    def month_sum_list(subdf): #可以把每一个group过后的东西都是一个子dataframe，函数里面完全按照dataframe操作即可
        sum_list = []
        for m in range(0,20):
            sum_list.append(subdf[subdf["month_No"]==m]["Trans_at"].sum())
        return sum_list

    month_sum_se = certid_grouped.apply(month_sum_list)
    month_sum_df = pd.DataFrame(month_sum_se,columns=['lists'])

    for m in range(0, 20):
        newcol = "month_sum_" + str(m)
        month_sum_df[newcol] = month_sum_df['lists'].apply(lambda x: x[m])

    month_sum_df = month_sum_df.drop("lists", axis=1, inplace=False)

####################次数##############################
    def month_cnt_list(subdf):  # 可以把每一个group过后的东西都是一个子dataframe，函数里面完全按照dataframe操作即可
        cnt_list = []
        for m in range(0, 20):
            cnt_list.append(subdf[subdf["month_No"] == m]["Trans_at"].count())
        return cnt_list

    month_cnt_se = certid_grouped.apply(month_cnt_list)
    month_cnt_df = pd.DataFrame(month_cnt_se, columns=['lists'])

    for m in range(0, 20):
        newcol = "month_cnt_" + str(m)
        month_cnt_df[newcol] = month_cnt_df['lists'].apply(lambda x: x[m])

    month_cnt_df = month_cnt_df.drop("lists", axis=1, inplace=False)
#########################################################
    month_sum_df.to_csv("month_sum_temp.csv")
    month_cnt_df.to_csv("month_cnt_temp.csv")


    ########################################################################################################
    agg_stat_df = pd.read_csv("agg_stat_temp.csv", sep=",", low_memory=False, error_bad_lines=False)
    month_sum_df = pd.read_csv("month_sum_temp.csv", sep=",", low_memory=False, error_bad_lines=False)
    month_cnt_df = pd.read_csv("month_cnt_temp.csv", sep=",", low_memory=False, error_bad_lines=False)

    train_certid_df = pd.read_csv("train_certid_date_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)
    test_certid_df = pd.read_csv("test_certid_date_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    certid_df = pd.concat([train_certid_df, test_certid_df], axis=0)
    certid_df["apply_dateNo"] = certid_df["first_time"].map(lambda x: DateDict[x])

    certid_dummies = pd.get_dummies(certid_df[["sex", "aera_code"]])

    certid_dummies = pd.concat([certid_df[['certid',"apply_dateNo"]], certid_dummies], axis=1)

    age_df = certid_df[['certid', 'age']]  # .to_frame()

    Effect_df = pd.merge(left=certid_dummies, right=age_df, how='left', left_on='certid', right_on='certid')

    Effect_df = pd.merge(left=Effect_df, right=agg_stat_df, how='left', left_on='certid', right_on='certid')

    Effect_df = pd.merge(left=Effect_df, right=month_sum_df, how='left', left_on='certid', right_on='certid')
    Effect_df = pd.merge(left=Effect_df, right=month_cnt_df, how='left', left_on='certid', right_on='certid')

    #Effect_df = Effect_df.fillna(-1)

    Effect_df["apply_max_delta"] = Effect_df.apply(lambda df:  getDelta(df, "dateNo-max", "apply_dateNo"), axis=1)
    Effect_df["apply_mean_delta"] = Effect_df.apply(lambda df: getDelta(df, "dateNo-mean", "apply_dateNo"), axis=1)
    Effect_df["min_apply_delta"] = Effect_df.apply(lambda df: getDelta(df, "apply_dateNo", "dateNo-min"), axis=1)

    Effect_df = pd.merge(left=Effect_df, right=label_df, how='left', left_on='certid', right_on='certid')

    #Effect_df = Effect_df.fillna(-1)

################################################Join 后再分析#######################################

    Effect_df = Effect_df.fillna(-1)
 ####################地区详细分析###############
    le = preprocessing.LabelEncoder()
    le.fit(Effect_df["aera_code"])
    Effect_df["aera_code_encode"] = le.transform(Effect_df["aera_code"])

    Good_df = Effect_df[Effect_df["label"] == 1]
    Bad_df = Effect_df[Effect_df["label"] == 0]
    Bad_items = Bad_df["aera_code"].value_counts().axes[0].values
    Good_items = Good_df["aera_code"].value_counts().axes[0].values
    risk_items = np.setdiff1d(Bad_items[:int(Bad_items.shape[0] * 0.2)], Good_items[:int(Bad_items.shape[0] * 0.2)])
    new_col = "is_risk_" + "aera_code"
    Effect_df[new_col] = Effect_df["aera_code"].map(lambda x: is_risk_items(x,risk_items))

    Effect_df["prov"] = Effect_df["aera_code"].apply(lambda x: str(x)[0:2])
    Effect_df["city"] = Effect_df["aera_code"].apply(lambda x: str(x)[2:4])
    Effect_df["county"] = Effect_df["aera_code"].apply(lambda x: str(x)[-2:])
###############################################

####################年龄分析################

    Effect_df["age_section"] = Effect_df["age"].apply(lambda x: age_section(x))



##########################消费稳定性分析###############################

    Effect_df["mean_money_month_list"] = Effect_df.apply(mean_money_month_list, axis=1)
    for m in range(0, 20):
        newcol = "mean_money_m" + str(m)
        Effect_df[newcol] = Effect_df["mean_money_month_list"].apply(lambda x: x[m])

    Effect_df = Effect_df.drop("mean_money_month_list", axis=1, inplace=False)




    Effect_df["has_trans_month"] = Effect_df.apply(has_trans_month, axis=1)
    Effect_df["trans_month_var"] = Effect_df.apply(lambda x : month_sum_var(x, r"month_No-cnt_"), axis=1)
    Effect_df["trans_month_max"] = Effect_df.apply(lambda x : month_sum_max(x, r"month_No-cnt_"), axis=1)
    Effect_df["trans_month_mean"] = Effect_df.apply(lambda x : month_sum_mean(x, r"month_No-cnt_"), axis=1)
    Effect_df["trans_month_p2p"] = Effect_df.apply(lambda x : month_sum_p2p(x, r"month_No-cnt_"), axis=1)
    Effect_df["trans_month_stable"] = Effect_df.apply(lambda x: month_sum_stable(x, r"month_No-cnt_"), axis=1)

    Effect_df["month_sum_var"] = Effect_df.apply(lambda x : month_sum_var(x, r"month_sum_"), axis=1)
    Effect_df["month_sum_max"] = Effect_df.apply(lambda x : month_sum_max(x, r"month_sum_"), axis=1)
    Effect_df["month_sum_mean"] = Effect_df.apply(lambda x : month_sum_mean(x, r"month_sum_"), axis=1)
    Effect_df["month_sum_p2p"] = Effect_df.apply(lambda x : month_sum_p2p(x, r"month_sum_"), axis=1)
    Effect_df["month_sum_stable"] = Effect_df.apply(lambda x: month_sum_stable(x, r"month_sum_"), axis=1)
    #print Effect_df["month_No-cnt_0"]
###############################################
    Effect_df.to_csv("agg_math_stable.csv",index=False)



