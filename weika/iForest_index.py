"""
oneclass for  TradingFraud
python 3.5
"""

#sklearn
from sklearn import  svm
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

#load  data, positive:negative=1:4
#df_All = pd.read_csv("Labeled_noNAN_1608_sample.csv", header=0, sep=',') 
#load  data, positive:negative=1:4
#df_All = pd.read_csv("index_label.csv", header=0, sep=',')   
df_All = pd.read_csv("201608_indexed_withlabel.csv", header=0, sep=',')   
df_All = shuffle(df_All)

 
df_All.columns = ["settle_tp_filled","settle_cycle_filled","block_id_filled","trans_fwd_st_filled","trans_rcv_st_filled","sms_dms_conv_in_filled","cross_dist_in_filled","tfr_in_in_filled","trans_md_filled","source_region_cd_filled","dest_region_cd_filled","cups_card_in_filled","cups_sig_card_in_filled","card_class_filled","card_attr_filled","acq_ins_tp_filled","fwd_ins_tp_filled","rcv_ins_tp_filled","iss_ins_tp_filled","acpt_ins_tp_filled","resp_cd1_filled","resp_cd2_filled","resp_cd3_filled","resp_cd4_filled","cu_trans_st_filled","sti_takeout_in_filled","trans_id_filled","trans_tp_filled","trans_chnl_filled","card_media_filled","card_brand_filled","trans_id_conv_filled","trans_curr_cd_filled","conn_md_filled","msg_tp_filled","msg_tp_conv_filled","trans_proc_cd_filled","trans_proc_cd_conv_filled","mchnt_tp_filled","pos_entry_md_cd_filled","card_seq_filled","pos_cond_cd_filled","pos_cond_cd_conv_filled","term_tp_filled","rsn_cd_filled","addn_pos_inf_filled","orig_msg_tp_filled","orig_msg_tp_conv_filled","related_trans_id_filled","related_trans_chnl_filled","orig_trans_id_filled","orig_trans_chnl_filled","orig_card_media_filled","spec_settle_in_filled","iss_ds_settle_in_filled","acq_ds_settle_in_filled","upd_in_filled","exp_rsn_cd_filled","pri_cycle_no_filled","corr_pri_cycle_no_filled","disc_in_filled","orig_disc_curr_cd_filled","fwd_settle_conv_rt_filled","rcv_settle_conv_rt_filled","fwd_settle_curr_cd_filled","rcv_settle_curr_cd_filled","sp_mchnt_cd_filled","acq_ins_id_cd_BK_filled","acq_ins_id_cd_RG_filled","fwd_ins_id_cd_BK_filled","fwd_ins_id_cd_RG_filled","rcv_ins_id_cd_BK_filled","rcv_ins_id_cd_RG_filled","iss_ins_id_cd_BK_filled","iss_ins_id_cd_RG_filled","related_ins_id_cd_BK_filled","related_ins_id_cd_RG_filled","acpt_ins_id_cd_BK_filled","acpt_ins_id_cd_RG_filled","settle_fwd_ins_id_cd_BK_filled","settle_fwd_ins_id_cd_RG_filled","settle_rcv_ins_id_cd_BK_filled","settle_rcv_ins_id_cd_RG_filled","acct_ins_id_cd_BK_filled","acct_ins_id_cd_RG_filled","label_filled"]
 
df_X = df_All # df_All[["settle_tp_filled","settle_cycle_filled","block_id_filled","trans_fwd_st_filled","trans_rcv_st_filled","sms_dms_conv_in_filled","cross_dist_in_filled","tfr_in_in_filled","trans_md_filled","source_region_cd_filled","dest_region_cd_filled","cups_card_in_filled","cups_sig_card_in_filled","card_class_filled","card_attr_filled","acq_ins_tp_filled","fwd_ins_tp_filled","rcv_ins_tp_filled","iss_ins_tp_filled","acpt_ins_tp_filled","resp_cd1_filled","resp_cd2_filled","resp_cd3_filled","resp_cd4_filled","cu_trans_st_filled","sti_takeout_in_filled","trans_id_filled","trans_tp_filled","trans_chnl_filled","card_media_filled","card_brand_filled","trans_id_conv_filled","trans_curr_cd_filled","conn_md_filled","msg_tp_filled","msg_tp_conv_filled","trans_proc_cd_filled","trans_proc_cd_conv_filled","mchnt_tp_filled","pos_entry_md_cd_filled","card_seq_filled","pos_cond_cd_filled","pos_cond_cd_conv_filled","term_tp_filled","rsn_cd_filled","addn_pos_inf_filled","orig_msg_tp_filled","orig_msg_tp_conv_filled","related_trans_id_filled","related_trans_chnl_filled","orig_trans_id_filled","orig_trans_chnl_filled","orig_card_media_filled","spec_settle_in_filled","iss_ds_settle_in_filled","acq_ds_settle_in_filled","upd_in_filled","exp_rsn_cd_filled","pri_cycle_no_filled","corr_pri_cycle_no_filled","disc_in_filled","orig_disc_curr_cd_filled","fwd_settle_conv_rt_filled","rcv_settle_conv_rt_filled","fwd_settle_curr_cd_filled","rcv_settle_curr_cd_filled","sp_mchnt_cd_filled","acq_ins_id_cd_BK_filled","acq_ins_id_cd_RG_filled","fwd_ins_id_cd_BK_filled","fwd_ins_id_cd_RG_filled","rcv_ins_id_cd_BK_filled","rcv_ins_id_cd_RG_filled","iss_ins_id_cd_BK_filled","iss_ins_id_cd_RG_filled","related_ins_id_cd_BK_filled","related_ins_id_cd_RG_filled","acpt_ins_id_cd_BK_filled","acpt_ins_id_cd_RG_filled","settle_fwd_ins_id_cd_BK_filled","settle_fwd_ins_id_cd_RG_filled","settle_rcv_ins_id_cd_BK_filled","settle_rcv_ins_id_cd_RG_filled","acct_ins_id_cd_BK_filled","acct_ins_id_cd_RG_filled"]]
df_y = df_All["label_filled"]

X_train,X_test,y_train,y_test=train_test_split(df_X,df_y,test_size=0.2)
#split X_train  to  normal and outliers

X_train_normal = X_train[X_train['label_filled'] == 0].drop("label_filled",axis=1, inplace=False)
#X_train_outliers = X_train[X_train['label_filled'] == 1].drop("label_filled",axis=1, inplace=False)
X_test = X_test.drop("label_filled",axis=1, inplace=False)
X_train = X_train.drop("label_filled",axis=1, inplace=False)

#print X_test_normal.size
  
print("Load data done.")
 
# fit the model
clf = IsolationForest(n_estimators=1000,contamination = 0.05, n_jobs=-1,bootstrap=True)
clf.fit(X_train)
#clf.fit(X_train_normal)

 
#predict
y_pred_train=clf.predict(X_train)
y_pred_test=clf.predict(X_test)

#change predict_labeks  (1:-1)->to (0,1)
y_pred_train=np.where(y_pred_train>0,0,1)
y_pred_test =np.where(y_pred_test>0,0,1)

#print result
print("train data classification report: ")
print(classification_report(y_train,y_pred_train))
print()
print("test data  classification report:")
print(classification_report(y_test,y_pred_test))
print()



