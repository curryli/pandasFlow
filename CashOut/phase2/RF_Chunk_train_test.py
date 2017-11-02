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



################################################# 
model_name = "FE_train_data.mdl"
#df_train = pd.read_csv("idx_new_08_del.csv", low_memory=False)


#df_train = pd.read_csv("FE_train_data.csv", low_memory=False)


reader = pd.read_csv("FE_train_data.csv", low_memory=False, iterator=True)
     
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
df_train = pd.concat(chunks, ignore_index=True)


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
print "test all data sucessfully in %s seconds." % delta_time

confusion_matrix=confusion_matrix(y_test,pred)
print  confusion_matrix


precision_p = float(confusion_matrix[1][1])/float((confusion_matrix[0][1] + confusion_matrix[1][1]))
recall_p = float(confusion_matrix[1][1])/float((confusion_matrix[1][0] + confusion_matrix[1][1]))
 
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 

F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)

print F1_Score
 
