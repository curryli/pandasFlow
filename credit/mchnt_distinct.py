# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
 
#导入随机森林算法库
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
import datetime
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

# train_ori_df = pd.read_csv("train_repalce.csv", sep=",", low_memory=False, error_bad_lines=False)
#
# test_ori_df = pd.read_csv("test_replace.csv", sep=",", low_memory=False, error_bad_lines=False)
#
# Trans_ori_df = pd.concat([train_ori_df, test_ori_df], axis=0)
#
# Trans_ori_df = Trans_ori_df[["mchnt_cd","card_accprt_nm_loc"]]
#
# Trans_ori_df = Trans_ori_df.drop_duplicates()
# Trans_ori_df.to_csv("mchnt_distinct.csv",index=False)


risk_MC_df = pd.read_csv("risk_MC.csv", sep=",", low_memory=False, error_bad_lines=False)

Mchnt_df = pd.read_csv("mchnt_distinct.csv", sep=",", low_memory=False, error_bad_lines=False)

df_All = pd.merge(left=risk_MC_df, right=Mchnt_df, how='left', left_on='mchnt_cd', right_on='mchnt_cd')
df_All.to_csv("risk_MC_detail.csv",index=False)