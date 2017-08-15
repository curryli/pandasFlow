# -*- coding: utf-8 -*-
import pandas as pd
 
import scipy  
import scipy.stats.mstats as mst
from scipy.stats import chisquare  
import numpy


df_All = pd.read_csv("201608_indexed_withlabel.csv", header=0, sep=',')  
 
df_All.columns = ["settle_tp_filled_idx","settle_cycle_filled_idx","block_id_filled_idx","trans_fwd_st_filled_idx","trans_rcv_st_filled_idx","sms_dms_conv_in_filled_idx","cross_dist_in_filled_idx","tfr_in_in_filled_idx","trans_md_filled_idx","source_region_cd_filled_idx","dest_region_cd_filled_idx","cups_card_in_filled_idx","cups_sig_card_in_filled_idx","card_class_filled_idx","card_attr_filled_idx","acq_ins_tp_filled_idx","fwd_ins_tp_filled_idx","rcv_ins_tp_filled_idx","iss_ins_tp_filled_idx","acpt_ins_tp_filled_idx","resp_cd1_filled_idx","resp_cd2_filled_idx","resp_cd3_filled_idx","resp_cd4_filled_idx","cu_trans_st_filled_idx","sti_takeout_in_filled_idx","trans_id_filled_idx","trans_tp_filled_idx","trans_chnl_filled_idx","card_media_filled_idx","card_brand_filled_idx","trans_id_conv_filled_idx","trans_curr_cd_filled_idx","conn_md_filled_idx","msg_tp_filled_idx","msg_tp_conv_filled_idx","trans_proc_cd_filled_idx","trans_proc_cd_conv_filled_idx","mchnt_tp_filled_idx","pos_entry_md_cd_filled_idx","card_seq_filled_idx","pos_cond_cd_filled_idx","pos_cond_cd_conv_filled_idx","term_tp_filled_idx","rsn_cd_filled_idx","addn_pos_inf_filled_idx","orig_msg_tp_filled_idx","orig_msg_tp_conv_filled_idx","related_trans_id_filled_idx","related_trans_chnl_filled_idx","orig_trans_id_filled_idx","orig_trans_chnl_filled_idx","orig_card_media_filled_idx","spec_settle_in_filled_idx","iss_ds_settle_in_filled_idx","acq_ds_settle_in_filled_idx","upd_in_filled_idx","exp_rsn_cd_filled_idx","pri_cycle_no_filled_idx","corr_pri_cycle_no_filled_idx","disc_in_filled_idx","orig_disc_curr_cd_filled_idx","fwd_settle_conv_rt_filled_idx","rcv_settle_conv_rt_filled_idx","fwd_settle_curr_cd_filled_idx","rcv_settle_curr_cd_filled_idx","sp_mchnt_cd_filled_idx","acq_ins_id_cd_BK_filled_idx","acq_ins_id_cd_RG_filled_idx","fwd_ins_id_cd_BK_filled_idx","fwd_ins_id_cd_RG_filled_idx","rcv_ins_id_cd_BK_filled_idx","rcv_ins_id_cd_RG_filled_idx","iss_ins_id_cd_BK_filled_idx","iss_ins_id_cd_RG_filled_idx","related_ins_id_cd_BK_filled_idx","related_ins_id_cd_RG_filled_idx","acpt_ins_id_cd_BK_filled_idx","acpt_ins_id_cd_RG_filled_idx","settle_fwd_ins_id_cd_BK_filled_idx","settle_fwd_ins_id_cd_RG_filled_idx","settle_rcv_ins_id_cd_BK_filled_idx","settle_rcv_ins_id_cd_RG_filled_idx","acct_ins_id_cd_BK_filled_idx","acct_ins_id_cd_RG_filled_idx","label_filled_idx"]

#df_X = [1, 0, 1, 0, 1, 0]
#df_y = [0, 1, 0, 1, 0, 1]
#print mst.chisquare(f_obs=df_X, f_exp=df_y)

df_y = numpy.array(df_All["label_filled_idx"])
#print df_y


for i in df_All.columns:
    df_X = numpy.array(df_All[i])
    #print df_X
    print scipy.stats.chisquare(f_obs=df_X, f_exp=df_y)[1]
    #print pd.crosstab(df_X, df_y, margins=True)
    
    #不能直接把df_X,df_y塞到scipy.stats.chisquare里面。好像得先针对df_X,df_y的每种取值的频率分别统计，然后再计算。

 