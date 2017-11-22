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
from dateutil import parser
from woe_pandas import WOE_pandas

# from month_cnt_func import Month_Cnt_class



#####################################

if __name__ == '__main__':
    mchnt_ana_df = pd.read_csv("mchnt_ana.csv", sep=",", low_memory=False, error_bad_lines=False)


    def Bad_ratio(x):
        return float(x["mchnt_Bad_cnt"]) / float(x["trans_cnt"])


    def risk_ratio(x):
        return float(x["mchnt_risk_cnt"]) / float(x["trans_cnt"])


    def Badcd_ratio(x):
        return float(x["mchntcd_Bad_cnt"]) / float(x["trans_cnt"])


    def riskcd_ratio(x):
        return float(x["mchntcd_risk_cnt"]) / float(x["trans_cnt"])



    mchnt_ana_df["Bad_mchnt_ratio"] = mchnt_ana_df.apply(Bad_ratio, axis=1)
    mchnt_ana_df["risk_mchnt_ratio"] = mchnt_ana_df.apply(risk_ratio, axis=1)
    mchnt_ana_df["Bad_mchntcd_ratio"] = mchnt_ana_df.apply(Badcd_ratio, axis=1)
    mchnt_ana_df["risk_mchntcd_ratio"] = mchnt_ana_df.apply(riskcd_ratio, axis=1)


    label_df = pd.read_csv("train_label_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)
    mchnt_ana_df = pd.merge(left=mchnt_ana_df, right=label_df, how='left', left_on='certid', right_on='certid')

    mchnt_ana_df= mchnt_ana_df.fillna(-1)


    mchnt_ana_df.to_csv("mchnt_stat.csv", index=False)