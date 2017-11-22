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


if __name__ == '__main__':
    #train_ori_df = pd.read_csv("check_ori.csv", sep=",", low_memory=False, error_bad_lines=False)

    #train_ori_df = pd.read_csv("train_trans_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)
    train_ori_df = pd.read_csv("train_spark.csv", sep=",", low_memory=False, error_bad_lines=False)

    #print train_ori_df.shape


    check_certid = pd.read_csv("checklst_train.csv", sep=",", low_memory=False, error_bad_lines=False)


    Trans_ori_df = pd.merge(left=train_ori_df, right=check_certid, how='inner',  left_on='certid', right_on='certid')

    Trans_ori_df = Trans_ori_df.fillna(-1)

    Trans_ori_df.to_csv("check_trans.csv",index=False)