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

class WOE_pandas(object):
    def __init__(self, df):
        self.df = df
        self.Grp_all = df["label"].value_counts()
        self.total_good = self.cnt_pair(self.Grp_all)[0]
        self.total_bad = self.cnt_pair(self.Grp_all)[1]


    def cnt_pair(self, gs):
        cnt_keys = gs.axes[0].values
        cnt_values = gs.values
        cnt_dict = {}
        for i in range(0, len(cnt_keys)):
            cnt_dict[cnt_keys[i]] = cnt_values[i]

        if cnt_dict.has_key(0):
            bad_cnt = cnt_dict[0]
        else:
            bad_cnt = 0

        if cnt_dict.has_key(1):
            good_cnt = cnt_dict[1]
        else:
            good_cnt = 0

        return (bad_cnt, good_cnt)

    def CalcWOE_nominal(self, VarName, label):  # 名义变量WOE
        WOE_Map = pd.DataFrame()
        Vars = np.unique(self.df[VarName])  # 该列的取值distinct数组
        for v in Vars:
            tmp = self.df[VarName] == v
            tmp_df = self.df[tmp]
            grp = tmp_df["label"].value_counts()
            Bad = self.cnt_pair(grp)[0]
            Good = self.cnt_pair(grp)[1]
            good_ratio = float(Good) / float(self.total_good)
            bad_ratio = float(Bad) / float(self.total_bad)
            if good_ratio ==0:
                good_ratio = 0.01
            if bad_ratio ==0:
                bad_ratio = 0.01
            WOE = np.log(bad_ratio / good_ratio)
            IV = (bad_ratio - good_ratio) * WOE
            result = pd.DataFrame([[VarName, v, WOE, IV]], index=None, columns=['variable', 'class', 'woe', 'iv'])
            WOE_Map = WOE_Map.append(result, ignore_index=True)
        return WOE_Map

    def CalcWOE_bin(self, VarName, label, N):  # 要保证该列数据是数值型。N是对数值型变量N等分
        WOE_Map = pd.DataFrame()
        max_value = max(self.df[VarName])
        min_value = min(self.df[VarName])
        bin = float(max_value - min_value) / N
        for i in range(N):
            bin_U = min_value + (i + 1) * bin
            bin_L = bin_U - bin
            if i == 1:
                tmp = (self.df[VarName] >= bin_L) & (self.df[VarName] <= bin_U)
                tmp_df = self.df[tmp]
                grp = tmp_df["label"].value_counts()
            else:
                tmp = (self.df[VarName] > bin_L) & (self.df[VarName] <= bin_U)
                tmp_df = self.df[tmp]
                grp = tmp_df["label"].value_counts()
            Bad = self.cnt_pair(grp)[0]
            Good = self.cnt_pair(grp)[1]
            good_ratio = float(Good) / float(self.total_good)
            bad_ratio = float(Bad) / float(self.total_bad)
            if good_ratio ==0:
                good_ratio = 0.01
            if bad_ratio ==0:
                bad_ratio = 0.01
            WOE = np.log(bad_ratio / good_ratio)
            IV = (bad_ratio - good_ratio) * WOE
            result = pd.DataFrame([[VarName, [bin_L, bin_U, WOE], WOE, IV]],
                                  index=None, columns=['variable', 'class+woe', 'woe', 'iv'])
            WOE_Map = WOE_Map.append(result, ignore_index=True)
        return WOE_Map

    def ReplaceWOE_nominal(self, VarName, VarWOE):
        dict1 = dict.fromkeys(VarWOE['class'])
        j = 0
        for key in dict1:
            dict1[key] = VarWOE['woe'][j]
            j = j + 1
        #self.df[VarName] = self.df[VarName].map(dict1)
        self.df[VarName] = self.df[VarName].map(lambda x: dict1[x])
        return self.df


    def ReplaceWOE_bin(self, VarName, VarWOE):
        items = np.unique(self.df[VarName])
        m = min(self.df[VarName])
        dict2 = {}
        for it in items:
            if it == m:
                dict2[it] = VarWOE['class+woe'][0][2]
            else:
                for l, u, w in VarWOE['class+woe']:
                    if (it > l) & (it <= u):
                        dict2[it] = w
        #self.df[VarName] = self.df[VarName].map(dict2)
        self.df[VarName] = self.df[VarName].map(lambda x: dict2[x])
        return self.df


    def AddWOE_nominal(self, VarName, VarWOE):
        dict1 = dict.fromkeys(VarWOE['class'])
        j = 0
        for key in dict1:
            dict1[key] = VarWOE['woe'][j]
            j = j + 1
        new_col = VarName + "_woe"
        self.df[new_col] = self.df[VarName].map(lambda x: dict1[x])
        return self.df


    def AddWOE_bin(self, VarName, VarWOE):
        items = np.unique(self.df[VarName])
        m = min(self.df[VarName])
        dict2 = {}
        for it in items:
            if it == m:
                dict2[it] = VarWOE['class+woe'][0][2]
            else:
                for l, u, w in VarWOE['class+woe']:
                    if (it > l) & (it <= u):
                        dict2[it] = w
        new_col = VarName + "_woe"
        self.df[new_col] = self.df[VarName].map(lambda x: dict2[x])
        return self.df


if __name__ == '__main__':
    train_ori_df = pd.read_csv("small_data.csv", sep=",", low_memory=False, error_bad_lines=False)
    #train_ori_df = pd.read_csv("train_trans_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)

    label_df = pd.read_csv("train_label_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)
    Trans_ori_df = pd.merge(left=train_ori_df, right=label_df, how='left', left_on='certid', right_on='certid')

    Trans_ori_df.loc[0:2, ['label']] = 1
    Trans_ori_df.loc[3:5, ['label']] = -1

    #print Trans_ori_df
    woe_pandas = WOE_pandas(Trans_ori_df)

    print "CalcWOE_nominal"
    WOE_nominal =  woe_pandas.CalcWOE_nominal('trans_chnl', 'label')

    print "CalcWOE_bin"
    WOE_bin = woe_pandas.CalcWOE_bin('Trans_at', 'label', 5)

    #print WOE_bin

    Trans_ori_df = woe_pandas.ReplaceWOE_nominal('trans_chnl', WOE_nominal)
    Trans_ori_df = woe_pandas.ReplaceWOE_bin('Trans_at', WOE_bin)






    ##################
    df_All = pd.read_csv("train_2.csv", sep=',')
    df_All = df_All.fillna(-1)
    print df_All.columns
    wp1 = WOE_pandas(df_All)

    WOE_nominal = wp1.CalcWOE_nominal('date-countDistinct', 'label')
    wp1.ReplaceWOE_nominal('date-countDistinct', WOE_nominal)


    WOE_bin = wp1.CalcWOE_bin('cdhd_at-mean', 'label', 10)
    print WOE_bin
    df_All = wp1.ReplaceWOE_bin('cdhd_at-mean', WOE_bin)

    print df_All.columns


