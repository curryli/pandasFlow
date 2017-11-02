# -*- coding: utf-8 -*-
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import shuffle  
from sklearn.externals import joblib 
import numpy as np
import datetime

start_time = datetime.datetime.now()

  
label='label' # label的值就是二元分类的输出
cardcol= 'pri_acct_no_conv'
time = "tfr_dt_tm"

model_name = "201609_new_FE.mdl"

################################################# 
#df_train = pd.read_csv("idx_new_08_del.csv", low_memory=False)
df_train = pd.read_csv("201609_new_FE.csv", low_memory=False)
 
x_columns = [x for x in df_train.columns if x not in [label,cardcol,time, "term_id", "mchnt_cd"]]
label='label' # 
cardcol= 'pri_acct_no_conv'
 
#############################################################
clf_load = joblib.load(model_name)
print "model loaded sucessfully."
################################################# 
#chunker_rawData = pd.read_csv("FE_db_1012.csv", low_memory=False, chunksize = 100000)
chunker_rawData = pd.read_csv("FE_test_real.csv", low_memory=False, chunksize = 100000)
#chunker_rawData = pd.read_csv("idx_new_08_del.csv", low_memory=False, chunksize = 1000)


y_test = np.array([1])
pred = np.array([1])

print type(y_test) 

i = 0
for df_test in chunker_rawData:
    y_tmp = df_test.label.values
    X_test =  df_test[x_columns] 
    pred_tmp = clf_load.predict(X_test)
    
    y_test = np.append(y_test,y_tmp)
    pred = np.append(pred,pred_tmp) 
    if (i%10)==0:
        print i 
    i = i+1
     
print y_test.shape, pred.shape
end_time = datetime.datetime.now()
delta_time = str((end_time-start_time).total_seconds())
print "test all data sucessfully in %s seconds." % delta_time

confusion_matrix=confusion_matrix(y_test,pred)
print  confusion_matrix


precision_p = float(confusion_matrix[1][1])/float((confusion_matrix[0][1] + confusion_matrix[1][1]))
recall_p = float(confusion_matrix[1][1])/float((confusion_matrix[1][0] + confusion_matrix[1][1]))
 
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 

F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)

print F1_Score
 
