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
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve
import numpy as np
  
label='label' # label的值就是二元分类的输出
cardcol= 'pri_acct_no_conv'
time = "tfr_dt_tm"

df_All = pd.read_csv("idx_weika_07.csv", sep=',') 
df_All = shuffle(df_All) 
print df_All.shape[1]
 
x_columns = [x for x in df_All.columns if x not in [label]]
print x_columns
df_X = df_All[x_columns]
df_y  = df_All[label]
 
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)
  
#n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
grd = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100,max_depth=10, min_samples_leaf =60, min_samples_split =1200, max_features=10, subsample=0.7, random_state=10)
  
  
card_train = X_train[cardcol]
card_test = X_test[cardcol]

X_train_T = X_train.drop(cardcol, axis = 1, inplace=False) 
X_test_T = X_test.drop(cardcol, axis = 1, inplace=False) 


'''使用X_train训练GBDT模型，后面用此模型构造特征'''
grd.fit(X_train_T, y_train)


gene_feature = grd.apply(X_train_T)[:, :, 0]

y_train_np = y_train.values.reshape((y_train.shape[0], 1)) 
train_new = np.hstack((X_train,gene_feature,y_train_np ))
print train_new.shape

y_test_np = y_test.values.reshape((y_test.shape[0], 1)) 
gene_feature_test = grd.apply(X_test_T)[:, :, 0]
test_new = np.hstack((X_test,gene_feature_test,y_test_np ))
print test_new.shape

newdata = pd.DataFrame(np.vstack((train_new,test_new)))
print newdata.shape

newdata.to_csv('weika_GBDT_07.csv',  index=False,header=False)