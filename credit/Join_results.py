# coding=utf-8
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score

from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import datetime
from collections import Counter
from xgboost.sklearn import XGBClassifier
import numpy as np

df_test = pd.read_csv("test_certid_date_encrypt.csv", sep=',')

certid_test_DF = df_test[["certid"]]
my_result = pd.read_csv("pred_label_right.csv", sep=',')

df_All = pd.merge(left=certid_test_DF, right=my_result, how='left', left_on='certid', right_on='certid')
df_All.fillna(-1)

df_All.to_csv("InnoDeep_1113.csv",index=False)


