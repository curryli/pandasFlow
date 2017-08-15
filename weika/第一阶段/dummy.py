# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
 
#导入随机森林算法库
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
 
  
df_All = pd.read_csv("Labeled_noNAN_1608.csv", header=0, sep=',')  
print df_All.shape[1]
 
df_All.columns = ["pri_acct_no_conv_filled","day_week_filled","hour_filled","tfr_dt_tm_filled","trans_at_filled","total_disc_at_filled","settle_tp_filled","settle_cycle_filled","block_id_filled","trans_fwd_st_filled","trans_rcv_st_filled","sms_dms_conv_in_filled","cross_dist_in_filled","tfr_in_in_filled","trans_md_filled","source_region_cd_filled","dest_region_cd_filled","cups_card_in_filled","cups_sig_card_in_filled","card_class_filled","card_attr_filled","acq_ins_tp_filled","fwd_ins_tp_filled","rcv_ins_tp_filled","iss_ins_tp_filled","acpt_ins_tp_filled","resp_cd1_filled","resp_cd2_filled","resp_cd3_filled","resp_cd4_filled","cu_trans_st_filled","sti_takeout_in_filled","trans_id_filled","trans_tp_filled","trans_chnl_filled","card_media_filled","trans_id_conv_filled","trans_curr_cd_filled","conn_md_filled","msg_tp_filled","msg_tp_conv_filled","trans_proc_cd_filled","trans_proc_cd_conv_filled","mchnt_tp_filled","pos_entry_md_cd_filled","pos_cond_cd_filled","pos_cond_cd_conv_filled","term_tp_filled","rsn_cd_filled","addn_pos_inf_filled","iss_ds_settle_in_filled","acq_ds_settle_in_filled","upd_in_filled","pri_cycle_no_filled","disc_in_filled","fwd_settle_conv_rt_filled","rcv_settle_conv_rt_filled","fwd_settle_curr_cd_filled","rcv_settle_curr_cd_filled","acq_ins_id_cd_BK_filled","acq_ins_id_cd_RG_filled","fwd_ins_id_cd_BK_filled","fwd_ins_id_cd_RG_filled","rcv_ins_id_cd_BK_filled","rcv_ins_id_cd_RG_filled","iss_ins_id_cd_BK_filled","iss_ins_id_cd_RG_filled","acpt_ins_id_cd_BK_filled","acpt_ins_id_cd_RG_filled","settle_fwd_ins_id_cd_BK_filled","settle_fwd_ins_id_cd_RG_filled","settle_rcv_ins_id_cd_BK_filled","settle_rcv_ins_id_cd_RG_filled","label_filled"] 
#df_All = df_All.loc[:3000]
 
df_dummies = pd.get_dummies(df_All[["settle_tp_filled","settle_cycle_filled","block_id_filled","trans_fwd_st_filled","trans_rcv_st_filled","sms_dms_conv_in_filled","cross_dist_in_filled","tfr_in_in_filled","trans_md_filled","source_region_cd_filled","dest_region_cd_filled","cups_card_in_filled","cups_sig_card_in_filled","card_class_filled","card_attr_filled","acq_ins_tp_filled","fwd_ins_tp_filled","rcv_ins_tp_filled","iss_ins_tp_filled","acpt_ins_tp_filled","resp_cd1_filled","resp_cd2_filled","resp_cd3_filled","resp_cd4_filled","cu_trans_st_filled","sti_takeout_in_filled","trans_id_filled","trans_tp_filled","trans_chnl_filled","card_media_filled","trans_id_conv_filled","trans_curr_cd_filled","conn_md_filled","msg_tp_filled","msg_tp_conv_filled","trans_proc_cd_filled","trans_proc_cd_conv_filled","mchnt_tp_filled","pos_entry_md_cd_filled","pos_cond_cd_filled","pos_cond_cd_conv_filled","term_tp_filled","rsn_cd_filled","addn_pos_inf_filled","iss_ds_settle_in_filled","acq_ds_settle_in_filled","upd_in_filled","pri_cycle_no_filled","disc_in_filled","fwd_settle_conv_rt_filled","rcv_settle_conv_rt_filled","fwd_settle_curr_cd_filled","rcv_settle_curr_cd_filled","acq_ins_id_cd_BK_filled","acq_ins_id_cd_RG_filled","fwd_ins_id_cd_BK_filled","fwd_ins_id_cd_RG_filled","rcv_ins_id_cd_BK_filled","rcv_ins_id_cd_RG_filled","iss_ins_id_cd_BK_filled","iss_ins_id_cd_RG_filled","acpt_ins_id_cd_BK_filled","acpt_ins_id_cd_RG_filled","settle_fwd_ins_id_cd_BK_filled","settle_fwd_ins_id_cd_RG_filled","settle_rcv_ins_id_cd_BK_filled","settle_rcv_ins_id_cd_RG_filled"]])  
print df_dummies.loc[:30]

#df_All = pd.concat([df_All[["pri_acct_no_conv_filled","day_week_filled","hour_filled","tfr_dt_tm_filled","trans_at_filled","total_disc_at_filled"]],df_dummies, df_All["label_filled"]], axis=1)



df_X = pd.concat([df_All[["day_week_filled","hour_filled","tfr_dt_tm_filled","trans_at_filled","total_disc_at_filled"]],df_dummies], axis=1)

#df_X.loc[0:200].to_csv("dummy_1608.csv")

df_y = df_All["label_filled"]

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.4)

clf = RandomForestClassifier()

clf = clf.fit(X_train, y_train)
  
print clf.score(X_test, y_test)

pred = clf.predict(X_test) 

confusion_matrix=confusion_matrix(y_test,pred)
print  confusion_matrix


precision = precision_score(y_test, pred, average='macro') 
recall = recall_score(y_test, pred, average='macro') 
print ("Precision:", precision) 
print ("Recall:", recall) 


precision = precision_score(y_test, pred, average="weighted") 
recall = recall_score(y_test, pred, average="weighted") 
print ("Precision:", precision) 
print ("Recall:", recall) 