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
 
#df = pd.read_csv("idx_new_08_del.csv")
df = pd.read_csv("idx_only_gbdt.csv") 


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
 
train_all,test_all=train_test_split(df_All, test_size=0.2)
 

y_train = train_all.label
y_test = test_all.label

X_train = train_all[x_columns] 
X_test =  test_all[x_columns] 
 

#n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features="auto",max_leaf_nodes=None, bootstrap=True)

clf = clf.fit(X_train, y_train)
  
print clf.score(X_test, y_test)

pred = clf.predict(X_test) 

confusion_matrix=confusion_matrix(y_test,pred)
print  confusion_matrix


precision_p = float(confusion_matrix[1][1])/float((confusion_matrix[0][1] + confusion_matrix[1][1]))
recall_p = float(confusion_matrix[1][1])/float((confusion_matrix[1][0] + confusion_matrix[1][1]))
 
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 



 
#[17429   149]
# [  133  6834]




print "COMPARE:::::::::::::::::::::::::::::::::::::::::::::::::"
 

y_test_kmeans = KMeans(n_clusters=10, precompute_distances=True, random_state=3).fit_predict(X_test)


test_df = test_all
  
#compare_idx = y_predict^y_test  #相同为0，不同为1
compare_idx = np.bitwise_xor(pred, y_test.values.astype(np.int32))

 
compare_df = pd.DataFrame(compare_idx, index=test_df.index)
print compare_idx.shape

y_kmeans = pd.DataFrame(y_test_kmeans, index=test_df.index)
print y_kmeans.shape

 
test_df['compare'] = compare_df

test_df['y_kmeans'] = y_kmeans


test_df.to_csv('compare_results_RF_gbdt.csv')