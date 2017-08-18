# -*- coding: utf-8 -*-
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import shuffle  
from sklearn.cross_validation import train_test_split

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

label='label' # label的值就是二元分类的输出
cardcol= 'pri_acct_no_conv'
time = "tfr_dt_tm"
 
df = pd.read_csv("idx_new_08_del.csv")
#df = pd.read_csv("idx_weika_07.csv") 


label='label' # 
cardcol= 'pri_acct_no_conv'

Fraud = df[df.label == 1]
Normal = df[df.label == 0]
print "Ori Fraud shape:", Fraud.shape, "Ori Normal shape:", Normal .shape

card_c_F = Fraud['pri_acct_no_conv'].drop_duplicates()#涉欺诈交易卡号

#未出现在欺诈样本中的正样本数据
fine_N=Normal[(~Normal['pri_acct_no_conv'].isin(card_c_F))]
print "True Fraud shape:", Fraud.shape, "True Normal shape:", fine_N.shape

df_All=pd.concat([Fraud,fine_N], axis = 0)


################################################################################


df_All=shuffle(df_All)
  
x_columns = [x for x in df_All.columns if x not in [label,cardcol,time]]
 
Train_all,test_all=train_test_split(df_All, test_size=0.2)

X_test =  test_all[x_columns]  
y_test = test_all.label

#########################################################
train_all,valid_all=train_test_split(Train_all, test_size=0.5)
###############################################################

 
y_train = train_all.label
y_valid = valid_all.label

X_train = train_all[x_columns] 
X_valid =  valid_all[x_columns] 
  
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features="auto",max_leaf_nodes=None, bootstrap=True)

clf = clf.fit(X_train, y_train)
  
print clf.score(X_valid, y_valid)

pred = clf.predict(X_valid) 
 
print "COMPARE:::::::::::::::::::::::::::::::::::::::::::::::::"
 
valid_df = valid_all
  
#compare_idx = y_predict^y_valid  #相同为0，不同为1
compare_idx = np.bitwise_xor(pred, y_valid.values.astype(np.int32))

 
compare_df = pd.DataFrame(compare_idx, index=valid_df.index)
print compare_idx.shape

valid_df['compare'] = compare_df
valid_wrong = valid_df[(valid_df["compare"]==1)]

wrong_T = valid_wrong
for i in range(0,100):
    valid_wrong=pd.concat([valid_wrong, wrong_T], axis = 0)
    
boosted=pd.concat([Train_all,valid_wrong], axis = 0)
###############################################################################

 
y_train = valid_all.label
y_valid = train_all.label

X_train = valid_all[x_columns] 
X_valid = train_all[x_columns] 
  
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features="auto",max_leaf_nodes=None, bootstrap=True)

clf = clf.fit(X_train, y_train)
  
print clf.score(X_valid, y_valid)

pred = clf.predict(X_valid) 
 
print "COMPARE:::::::::::::::::::::::::::::::::::::::::::::::::"
 
valid_df = valid_all
  
#compare_idx = y_predict^y_valid  #相同为0，不同为1
compare_idx = np.bitwise_xor(pred, y_valid.values.astype(np.int32))

 
compare_df = pd.DataFrame(compare_idx, index=valid_df.index)
print compare_idx.shape

valid_df['compare'] = compare_df
valid_wrong = valid_df[(valid_df["compare"]==1)]

wrong_T = valid_wrong
for i in range(0,100):
    valid_wrong=pd.concat([valid_wrong, wrong_T], axis = 0)
    
boosted=pd.concat([boosted,valid_wrong], axis = 0)
 
###############################################################################

    
boosted = shuffle(boosted)

X_boosted = boosted[x_columns] 
y_boosted = boosted.label

#######################################


 

clf = clf.fit(X_boosted, y_boosted)
  
print clf.score(X_test, y_test)

pred = clf.predict(X_test) 

confusion_matrix_2=confusion_matrix(y_test,pred)
print  confusion_matrix_2


precision_p = float(confusion_matrix_2[1][1])/float((confusion_matrix_2[0][1] + confusion_matrix_2[1][1]))
recall_p = float(confusion_matrix_2[1][1])/float((confusion_matrix_2[1][0] + confusion_matrix_2[1][1]))
 
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 


#[[17238   149]
# [  122  7036]]
#('Precision:', 0.9792623521224774)
#('Recall:', 0.9829561329980442)