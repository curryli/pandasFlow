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
  
df_All = pd.read_csv("idx_new_08_del.csv", sep=',') 
df_All = shuffle(df_All) 
print df_All.shape[1]
 
#df_All.columns = ["pri_acct_no_conv","tfr_dt_tm","day_week","hour","trans_at","total_disc_at","settle_tp","settle_cycle","block_id","trans_fwd_st","trans_rcv_st","sms_dms_conv_in","cross_dist_in","tfr_in_in","trans_md","source_region_cd","dest_region_cd","cups_card_in","cups_sig_card_in","card_class","card_attr","acq_ins_tp","fwd_ins_tp","rcv_ins_tp","iss_ins_tp","acpt_ins_tp","resp_cd1","resp_cd2","resp_cd3","resp_cd4","cu_trans_st","sti_takeout_in","trans_id","trans_tp","trans_chnl","card_media","trans_id_conv","trans_curr_cd","conn_md","msg_tp","msg_tp_conv","trans_proc_cd","trans_proc_cd_conv","mchnt_tp","pos_entry_md_cd","pos_cond_cd","pos_cond_cd_conv","term_tp","rsn_cd","addn_pos_inf","iss_ds_settle_in","acq_ds_settle_in","upd_in","pri_cycle_no","disc_in","fwd_settle_conv_rt","rcv_settle_conv_rt","fwd_settle_curr_cd","rcv_settle_curr_cd","acq_ins_id_cd_BK","acq_ins_id_cd_RG","fwd_ins_id_cd_BK","fwd_ins_id_cd_RG","rcv_ins_id_cd_BK","rcv_ins_id_cd_RG","iss_ins_id_cd_BK","iss_ins_id_cd_RG","acpt_ins_id_cd_BK","acpt_ins_id_cd_RG","settle_fwd_ins_id_cd_BK","settle_fwd_ins_id_cd_RG","settle_rcv_ins_id_cd_BK","settle_rcv_ins_id_cd_RG","label"]#df_All = df_All.loc[:3000]
 
df_dummies =  df_All[["settle_tp","settle_cycle","block_id","trans_fwd_st","sms_dms_conv_in","cross_dist_in","tfr_in_in","trans_md","source_region_cd","dest_region_cd","cups_card_in","cups_sig_card_in","card_class","card_attr","acq_ins_tp","fwd_ins_tp","rcv_ins_tp","iss_ins_tp","acpt_ins_tp","resp_cd1","resp_cd3","cu_trans_st","sti_takeout_in","trans_id","trans_tp","trans_chnl","card_media","trans_curr_cd","conn_md","msg_tp","msg_tp_conv","trans_proc_cd","trans_proc_cd_conv","mchnt_tp","pos_entry_md_cd","pos_cond_cd","pos_cond_cd_conv","term_tp","rsn_cd","addn_pos_inf","iss_ds_settle_in","acq_ds_settle_in","upd_in","pri_cycle_no","disc_in","fwd_settle_conv_rt","fwd_settle_curr_cd","rcv_settle_curr_cd","acq_ins_id_cd_BK","acq_ins_id_cd_RG","fwd_ins_id_cd_BK","fwd_ins_id_cd_RG","rcv_ins_id_cd_BK","rcv_ins_id_cd_RG","iss_ins_id_cd_BK","iss_ins_id_cd_RG","acpt_ins_id_cd_BK","acpt_ins_id_cd_RG","settle_fwd_ins_id_cd_BK","settle_fwd_ins_id_cd_RG","settle_rcv_ins_id_cd_BK","settle_rcv_ins_id_cd_RG"]]

#print df_dummies.loc[:30]

#df_All = pd.concat([df_All[["pri_acct_no_conv_filled","day_week_filled","hour_filled","tfr_dt_tm_filled","trans_at_filled","total_disc_at_filled"]],df_dummies, df_All["label_filled"]], axis=1)



df_X = pd.concat([df_All[["tfr_dt_tm","day_week","hour","trans_at","total_disc_at"]],df_dummies], axis=1)
#df_X = df_All[["hour","trans_at","total_disc_at"]]
  

df_y = df_All["label"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)

  
#n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
grd = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100,max_depth=10, min_samples_leaf =60, min_samples_split =1200, max_features=10, subsample=0.7, random_state=10)
  
grd_enc = OneHotEncoder()
# 调用LR分类模型。
lr = LogisticRegression()

'''使用X_train训练GBDT模型，后面用此模型构造特征'''
grd.fit(df_X, df_y)


gene_feature = grd.apply(X_train)[:, :, 0]
X_train_new = np.hstack((X_train,gene_feature))
print X_train_new.shape

gene_feature_test = grd.apply(X_test)[:, :, 0]
X_test_new = np.hstack((X_test,gene_feature_test))
print X_test_new.shape

 
''' 
使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
'''
lr.fit(gene_feature, y_train)
# 用训练好的LR模型多X_test做预测

 

pred2 = lr.predict(gene_feature_test)

print "y_test shape: ", y_test.shape
print "predict shape: ", pred2.shape


confusion_matrix_2=confusion_matrix(y_test,pred2)
print  confusion_matrix_2


precision_p2 = float(confusion_matrix_2[1][1])/float((confusion_matrix_2[0][1] + confusion_matrix_2[1][1]))
recall_p2 = float(confusion_matrix_2[1][1])/float((confusion_matrix_2[1][0] + confusion_matrix_2[1][1]))
 
print ("Precision:", precision_p2) 
print ("Recall:", recall_p2) 


 