# -*- coding: utf-8 -*-
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import shuffle  
from sklearn.externals import joblib 
import numpy as np
import datetime
from sklearn.cross_validation import train_test_split

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score

from sklearn.metrics import precision_recall_fscore_support
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

start_time = datetime.datetime.now()


################################################# 
model_name = "FE_train_data.mdl"

reader = pd.read_csv("new_FE_idx.csv", low_memory=False, iterator=True)
     
loop = True
chunkSize = 100000 
chunks = []
i = 0
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
        if (i%5)==0:
            print i
        i = i+1
   
    except StopIteration:
        loop = False
        print "Iteration is stopped."
df_All = pd.concat(chunks, ignore_index=True)
print df_All.columns
df_All = df_All.drop(["Trans_at"], axis=1,inplace=False)
df_All = shuffle(df_All)


df_X = df_All.iloc[:, 3:]

df_y = df_All["label"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)
################################################
 
x_columns =  df_X.columns

clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)

print "start training"
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

cm1=confusion_matrix(y_test,pred)
print  cm1



print "Each class\n"
result = precision_recall_fscore_support(y_test,pred)
#print result
precision_0 = result[0][0]
recall_0 = result[1][0]
f1_0 = result[2][0]
precision_1 = result[0][1]
recall_1 = result[1][1]
f1_1 = result[2][1]
print "precision_0: ", precision_0,"  recall_0: ", recall_0, "  f1_0: ", f1_0



end_time = datetime.datetime.now()
delta_time = str((end_time-start_time).total_seconds())

