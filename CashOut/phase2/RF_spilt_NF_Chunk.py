# -*- coding: utf-8 -*-
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import shuffle  
from sklearn.externals import joblib 
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

start_time = datetime.datetime.now()

  
label='label' # label的值就是二元分类的输出
cardcol= 'pri_acct_no_conv'
time = "tfr_dt_tm"



################################################# 
model_name = "FE_train_data.mdl"
#df_train = pd.read_csv("idx_new_08_del.csv", low_memory=False)


#df_train = pd.read_csv("FE_train_data.csv", low_memory=False)

df_normal_test = pd.read_csv("FE_testNormal.csv", low_memory=False)
df_normal_train = pd.read_csv("FE_trainNormal.csv", low_memory=False)
df_fraud_all = pd.read_csv("FE_Fraud.csv", low_memory=False)

#trainNormal: ,891467,testNormal: ,28889678
      
print df_fraud_all.shape[0]   #16962

df_fraud_all=shuffle(df_fraud_all)

train_fraud,test_fraud = train_test_split(df_fraud_all, test_size=0.0075)

print train_fraud.shape[0],  test_fraud.shape[0]
#16834 128   230000:1

df_train = df_normal_train.concat([train_fraud], axis=0)

############################################################################
label='label' # 
cardcol= 'pri_acct_no_conv'
 
train_all=shuffle(df_train)
################################################
 
x_columns = [x for x in df_train.columns if x not in [label,cardcol,time, "term_id", "mchnt_cd"]]
  
y_train = train_all.label
 
X_train = train_all[x_columns]
 
#n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=10, max_features="auto",max_leaf_nodes=None, bootstrap=True)

print "start training"
clf = clf.fit(X_train, y_train)

end_time = datetime.datetime.now()
delta_time = str((end_time-start_time).total_seconds())

joblib.dump(clf, model_name)
print "model %s saved sucessfully in %s seconds." % (model_name, delta_time)

clf_load = joblib.load(model_name)
print "model loaded sucessfully."
################################################# 
#chunker_rawData = pd.read_csv("FE_db_1012.csv", low_memory=False, chunksize = 100000)
chunker_rawData = pd.read_csv("FE_test_data_new.csv", low_memory=False, chunksize = 100000)
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
print "test all normal data sucessfully in %s seconds." % delta_time




y_test_fraud = test_fraud.label.values
X_test_fraud =  test_fraud[x_columns] 
pred_test_fraud = clf_load.predict(X_test_fraud)
 
y_test = np.append(y_test,y_test_fraud)
pred = np.append(pred,pred_test_fraud) 

print y_test.shape, pred.shape
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
 
