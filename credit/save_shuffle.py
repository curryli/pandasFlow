# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score

# 导入随机森林算法库
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier

# df_All = pd.read_csv("train.csv", sep=',')
df_All = pd.read_csv("train_new.csv", sep=',')

df_All = df_All[(df_All["label"] == 0) | (df_All["label"] == 1)]

df_All = df_All.fillna(-1)

df_All = shuffle(df_All)

df_All.to_csv("train_shuffled.csv", index=False)

