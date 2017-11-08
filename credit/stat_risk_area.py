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


if __name__ == '__main__':
    df = pd.DataFrame({'animal': 'cat dog cat fish dog cat cat'.split(),
                       'size': list('SSMMMLL'),
                       'weight': [8, 10, 11, 1, 20, 12, 12],
                       'adult': [False] * 5 + [True] * 2})
    print df

    group = df.groupby("animal").apply(lambda subf: subf['size'][subf['weight'].idxmax()])
    print group

    #Effect_df = pd.read_csv("train_2.csv", sep=',')


    # # Good_df = df_All[df_All["label"] == 1]
    # # Bad_df = df_All[df_All["label"] == 0]
    #
    # Effect_df["prov"] = Effect_df["aera_code"].apply(lambda x: str(x)[0:2])
    # Effect_df["city"] = Effect_df["aera_code"].apply(lambda x: str(x)[-2:])
    # print Effect_df[["aera_code","prov","city"]]






