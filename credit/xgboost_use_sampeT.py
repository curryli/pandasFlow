# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score

# 导入随机森林算法库
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from xgboost.sklearn import XGBClassifier

df_All = pd.read_csv("agg_math_new.csv", sep=',')

df_All_stat_0 = pd.read_csv("agg_cat.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_0, how='left', left_on='certid', right_on='certid')


df_All_stat = pd.read_csv("translabel_stat.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat, how='left', left_on='certid', right_on='certid')

df_All_stat_2 = pd.read_csv("count_label.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_2, how='left', left_on='certid', right_on='certid')

df_All_stat_3 = pd.read_csv("count_label_isnot.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_3, how='left', left_on='certid', right_on='certid')

df_All_stat_4 = pd.read_csv("groupstat_2.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_4, how='left', left_on='certid', right_on='certid')

df_All_stat_5 = pd.read_csv("addition_stat_1.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_5, how='left', left_on='certid', right_on='certid')

df_All_stat_6 = pd.read_csv("groupMCC.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_6, how='left', left_on='certid', right_on='certid')

df_All_stat_7 = pd.read_csv("addition_stat_3.csv", sep=',')  #把 addition_stat_2.csv  里面一些过拟合的标记去掉
df_All = pd.merge(left=df_All, right=df_All_stat_7, how='left', left_on='certid', right_on='certid')

# df_All_stat_8 = pd.read_csv("MCC_detail.csv", sep=',')
# df_All = pd.merge(left=df_All, right=df_All_stat_8, how='left', left_on='certid', right_on='certid')


##########################
# df_All_stat_9 = pd.read_csv("mchnt_ana.csv", sep=',')
# df_All = pd.merge(left=df_All, right=df_All_stat_9, how='left', left_on='certid', right_on='certid')
#########################

label_df = pd.read_csv("train_label_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)
df_All = pd.merge(left=df_All, right=label_df, how='left', left_on='certid', right_on='certid')

df_All = df_All.fillna(-1)

#df_All = df_All.drop(["month_sum_0","pos_entry_md_cd_filled_idx-max","pos_cond_cd_filled_idx-var","day3_tot_cnt-sum","day3_tot_cnt-cnt_1","trans_chnl_filled_idx-max","mcc_cd_filled_idx-var","1hour_failure_cnt-sum","day3_tot_cnt-cnt_0","is_PW_need-cnt_0","is_Mchnt_changed-cnt_0","no_auth_id_resp_cd-sum","month_No-cnt_2","rcv_settle_at-max","hour-var","dateNo-var","trans_st_filled_idx-mean","resp_cd_filled_idx-sum","money_eq_last-cnt_1","min15_failure_cnt-sum","rcv_settle_at-sum","Trans_at-sum","Trans_at-peak_to_peak","card_media_cd_filled_idx-most_frequent_cnt","card_media_cd_filled_idx-max","iss_ins_cd_filled_idx-most_frequent_cnt","hist_query_cnt-mean","is_HR_CA_2_cnt_y","RMB_1_cnt","ws_mcc_cnt","month_No-cnt_12","Trans_at-max","card_attr_cd_filled_idx-sum","trans_st_filled_idx-std","resp_cd_filled_idx-max","mcc_cd_filled_idx-most_frequent_cnt","is_weekend-sum","2hour_failure_cnt-cnt_1","money_near_last-sum","money_near_last-cnt_0","is_lowrisk_MCC-cnt_0","RMB_7_cnt","date-most_frequent_cnt","month_sum_p2p","is_bigRMB_1000-sum","min15_failure_cnt-cnt_1","is_HR_MC_cnt_x","zhix_mcc_cnt","sex_male","month_No-cnt_11","fwd_settle_at-var","hour-countDistinct","month-min","mean_money_m13","has_trans_month","trans_month_max","count_89-cnt_0","is_Mchnt_changed-cnt_1","money_eq_last-sum","is_mcc_changed-sum","weekday-min","stageInMonth-most_frequent_item","trans_id_cd_filled_idx-countDistinct","pos_entry_md_cd_filled_idx-countDistinct","pos_entry_md_cd_filled_idx-peak_to_peak","is_HR_Term_2_cnt_y","jt_mcc_cnt","month_No-cnt_1","month-countDistinct","month-most_frequent_cnt","month-peak_to_peak","date-countDistinct","month_sum_14","trans_id_cd_filled_idx-median","trans_chnl_filled_idx-peak_to_peak","is_weekend-cnt_0","HR_risk8","rcv_settle_at-peak_to_peak","pos_cond_cd_filled_idx-most_frequent_cnt","orig_trans_st_filled_idx-mean","is_large_integer-sum","2hour_failure_cnt-sum","day7_no_trans-cnt_0","more_RSCR_cnt","LR_WK_med","month_No-cnt_4","month_sum_13","trans_chnl_filled_idx-countDistinct","1hour_no_trans-sum","day30_highrisk_MCC_cnt-cnt_0","cardholder_fail-sum","cardholder_fail-cnt_0","is_Night-cnt_0","is_HR_MC_2_cnt_y","is_HR_mcc_cnt_y","sy_mcc_cnt","dateNo-most_frequent_cnt","month_cnt_8","card_attr_cd_filled_idx-most_frequent_cnt","pos_entry_md_cd_filled_idx-most_frequent_cnt","iss_ins_cd_filled_idx-var","1hour_no_trans-cnt_0","is_bigRMB_500-cnt_0","is_bigRMB_500-cnt_1","is_HR_IIC_cnt_y","prov","card_attr_cd_filled_idx-var","trans_id_cd_filled_idx-peak_to_peak","orig_trans_st_filled_idx-std","cdhd_curr_cd_filled_idx-mean","is_PW_need-cnt_1","is_success-cnt_0","is_spec_airc-cnt_0","cur_highrisk_MCC_cnt-sum","gg_mcc_cnt","zs_mcc_cnt","dateNo-countDistinct","weekday-max","card_no-countDistinct_x","stageInMonth-median","month_cnt_9","trans_chnl_filled_idx-most_frequent_item","is_risk_iss_ins_cd_filled_idx-sum","resp_cd_filled_idx-most_frequent_cnt","resp_cd_filled_idx-peak_to_peak","rcv_settle_conv_rt_filled_idx-max","1hour_failure_cnt-cnt_1","is_success-sum","money_eq_last-cnt_0","1hour_highrisk_MCC_cnt-cnt_1","day30_no_trans-cnt_0","day30_no_trans-cnt_1","min15_failure_cnt-cnt_0","LH_city","xx_mcc_cnt","HR_month_MFC","month_cnt_4","mean_money_m14","trans_id_cd_filled_idx-most_frequent_cnt","pos_entry_md_cd_filled_idx-min","pos_entry_md_cd_filled_idx-median","trans_chnl_filled_idx-min","trans_chnl_filled_idx-median","cdhd_curr_cd_filled_idx-std","1hour_failure_cnt-cnt_0","is_lowrisk_MCC-cnt_1","cardholder_fail-cnt_1","min15_highrisk_MCC_cnt-cnt_1","cur_highrisk_MCC_cnt-cnt_0","is_HR_CA_cnt_y","RMB_9_cnt","month_No-cnt_0","month_cnt_11","mean_money_m15","trans_id_cd_filled_idx-most_frequent_item","is_risk_mchnt_cd_filled_idx-sum","resp_cd_filled_idx-countDistinct","is_large_integer-cnt_0","is_highrisk_MCC-sum","2hour_failure_cnt-cnt_0","day7_no_trans-cnt_1","not_HR_RSCR_cnt","fw_mcc_cnt","LR_RC_MFI","month_No-cnt_13","weekday-most_frequent_cnt","month_cnt_6","month_cnt_10","month_sum_mean","trans_id_cd_filled_idx-min","is_risk_pos_cond_cd_filled_idx-sum","trans_st_filled_idx-var","pos_entry_md_cd_filled_idx-most_frequent_item","card_media_cd_filled_idx-peak_to_peak","resp_cd_filled_idx-median","is_weekend-cnt_1","no_auth_id_resp_cd-cnt_1","not_HR_CA_cnt","not_HR_CA_2_cnt","rcv_settle_at-var","weekday-peak_to_peak","Trans_at-var","month_cnt_1","month_cnt_5","month_cnt_12","trans_st_filled_idx-most_frequent_cnt","card_media_cd_filled_idx-min","pos_cond_cd_filled_idx-countDistinct","resp_cd_filled_idx-most_frequent_item","iss_ins_cd_filled_idx-countDistinct","rcv_settle_conv_rt_filled_idx-mean","is_bigRMB_1000-cnt_0","is_bigRMB_1000-cnt_1","is_highrisk_MCC-cnt_0","hist_no_trans-cnt_0","hist_no_trans-cnt_1","1hour_no_trans-cnt_1","min15_highrisk_MCC_cnt-sum","1hour_highrisk_MCC_cnt-sum","not_HR_MC_cnt","LR_MA_delta","risk_mcc_cnt","zb_mcc_cnt","weekday-countDistinct","stageInMonth-peak_to_peak","month_cnt_0","month_cnt_7","trans_month_p2p","card_attr_cd_filled_idx-most_frequent_item","card_attr_cd_filled_idx-median","card_media_cd_filled_idx-countDistinct","card_media_cd_filled_idx-most_frequent_item","pos_cond_cd_filled_idx-min","pos_cond_cd_filled_idx-max","fwd_settle_conv_rt_filled_idx-mean","rcv_settle_conv_rt_filled_idx-var","is_spec_airc-cnt_1","is_HR_MC_cnt_y","not_HR_MC_2_cnt","RMB_bk_4_cnt","LR_date_freq","fdc_mcc_cnt","HR_TC_MFI","HR_risk4","HR_month_PP","month_No-cnt_14","cdhd_at-mean","cdhd_at-std","Trans_at-count","stageInMonth-countDistinct","stageInMonth-max","month_sum_15","month_cnt_2","month_cnt_3","month_cnt_13","month_cnt_14","month_cnt_16","age_section","card_attr_cd_filled_idx-max","trans_st_filled_idx-countDistinct","trans_st_filled_idx-sum","trans_curr_cd_filled_idx-max","pos_cond_cd_filled_idx-most_frequent_item","pos_cond_cd_filled_idx-median","fwd_settle_conv_rt_filled_idx-most_frequent_cnt","rcv_settle_conv_rt_filled_idx-sum","rcv_settle_conv_rt_filled_idx-peak_to_peak","cdhd_curr_cd_filled_idx-most_frequent_cnt","is_large_integer-cnt_1","is_success-cnt_1","is_norm_rate-mean","min15_highrisk_MCC_cnt-cnt_0","is_Night-cnt_1","is_mcc_changed-cnt_1","1hour_highrisk_MCC_cnt-cnt_0","is_HR_Term_cnt_x","HR_Term_ratio","HR_RSCR_ratio","LH_date_cd","wl_mcc_cnt","jy_mcc_cnt","HR_gg_mcc","month_No-cnt_15","month_No-cnt_16","month_No-cnt_17","month_No-cnt_18","month_No-cnt_19","month_No-cnt_20","fund_shortage-sum","cdhd_at-min","cdhd_at-max","cdhd_at-sum","cdhd_at-median","cdhd_at-var","cdhd_at-peak_to_peak","stageInMonth-min","month_sum_16","month_sum_17","month_sum_18","month_sum_19","month_cnt_15","month_cnt_17","month_cnt_18","month_cnt_19","mean_money_m16","mean_money_m17","mean_money_m18","mean_money_m19","trans_month_mean","card_attr_cd_filled_idx-countDistinct","card_attr_cd_filled_idx-min","card_attr_cd_filled_idx-peak_to_peak","is_risk_cdhd_curr_cd_filled_idx-sum","is_risk_fwd_settle_cruu_cd_filled_idx-sum","trans_st_filled_idx-most_frequent_item","trans_st_filled_idx-min","trans_st_filled_idx-max","trans_st_filled_idx-median","trans_st_filled_idx-peak_to_peak","is_risk_orig_trans_st_filled_idx-sum","is_risk_cdhd_conv_rt_filled_idx-sum","is_risk_card_accprt_nm_loc_filled_idx-sum","fwd_settle_cruu_cd_filled_idx-countDistinct","fwd_settle_cruu_cd_filled_idx-most_frequent_item","fwd_settle_cruu_cd_filled_idx-most_frequent_cnt","fwd_settle_cruu_cd_filled_idx-min","fwd_settle_cruu_cd_filled_idx-max","fwd_settle_cruu_cd_filled_idx-mean","fwd_settle_cruu_cd_filled_idx-sum","fwd_settle_cruu_cd_filled_idx-median","fwd_settle_cruu_cd_filled_idx-std","fwd_settle_cruu_cd_filled_idx-var","fwd_settle_cruu_cd_filled_idx-peak_to_peak","trans_curr_cd_filled_idx-countDistinct","trans_curr_cd_filled_idx-most_frequent_item","trans_curr_cd_filled_idx-most_frequent_cnt","trans_curr_cd_filled_idx-min","trans_curr_cd_filled_idx-mean","trans_curr_cd_filled_idx-sum","trans_curr_cd_filled_idx-median","trans_curr_cd_filled_idx-std","trans_curr_cd_filled_idx-var","trans_curr_cd_filled_idx-peak_to_peak","is_risk_trans_st_filled_idx-sum","rcv_settle_curr_cd_filled_idx-countDistinct","rcv_settle_curr_cd_filled_idx-most_frequent_item","rcv_settle_curr_cd_filled_idx-most_frequent_cnt","rcv_settle_curr_cd_filled_idx-min","rcv_settle_curr_cd_filled_idx-max","rcv_settle_curr_cd_filled_idx-mean","rcv_settle_curr_cd_filled_idx-sum","rcv_settle_curr_cd_filled_idx-median","rcv_settle_curr_cd_filled_idx-std","rcv_settle_curr_cd_filled_idx-var","rcv_settle_curr_cd_filled_idx-peak_to_peak","is_risk_fwd_settle_conv_rt_filled_idx-sum","card_no-countDistinct_y","is_risk_mcc_cd_filled_idx-sum","cdhd_conv_rt_filled_idx-countDistinct","cdhd_conv_rt_filled_idx-most_frequent_item","cdhd_conv_rt_filled_idx-most_frequent_cnt","cdhd_conv_rt_filled_idx-min","cdhd_conv_rt_filled_idx-max","cdhd_conv_rt_filled_idx-mean","cdhd_conv_rt_filled_idx-sum","cdhd_conv_rt_filled_idx-median","cdhd_conv_rt_filled_idx-std","cdhd_conv_rt_filled_idx-var","cdhd_conv_rt_filled_idx-peak_to_peak","card_media_cd_filled_idx-median","is_risk_trans_curr_cd_filled_idx-sum","is_risk_card_media_cd_filled_idx-sum","is_risk_rcv_settle_conv_rt_filled_idx-sum","is_risk_trans_id_cd_filled_idx-sum","pos_cond_cd_filled_idx-peak_to_peak","fwd_settle_conv_rt_filled_idx-countDistinct","fwd_settle_conv_rt_filled_idx-most_frequent_item","fwd_settle_conv_rt_filled_idx-min","fwd_settle_conv_rt_filled_idx-max","fwd_settle_conv_rt_filled_idx-sum","fwd_settle_conv_rt_filled_idx-median","fwd_settle_conv_rt_filled_idx-std","fwd_settle_conv_rt_filled_idx-var","fwd_settle_conv_rt_filled_idx-peak_to_peak","auth_id_resp_cd_filled_idx-most_frequent_item","resp_cd_filled_idx-min","orig_trans_st_filled_idx-countDistinct","orig_trans_st_filled_idx-most_frequent_item","orig_trans_st_filled_idx-most_frequent_cnt","orig_trans_st_filled_idx-min","orig_trans_st_filled_idx-max","orig_trans_st_filled_idx-sum","orig_trans_st_filled_idx-median","orig_trans_st_filled_idx-var","orig_trans_st_filled_idx-peak_to_peak","rcv_settle_conv_rt_filled_idx-countDistinct","rcv_settle_conv_rt_filled_idx-most_frequent_item","rcv_settle_conv_rt_filled_idx-most_frequent_cnt","rcv_settle_conv_rt_filled_idx-min","rcv_settle_conv_rt_filled_idx-median","rcv_settle_conv_rt_filled_idx-std","is_risk_resp_cd_filled_idx-sum","is_risk_card_attr_cd_filled_idx-sum","is_risk_rcv_settle_curr_cd_filled_idx-sum","is_risk_pos_entry_md_cd_filled_idx-sum","cdhd_curr_cd_filled_idx-countDistinct","cdhd_curr_cd_filled_idx-most_frequent_item","cdhd_curr_cd_filled_idx-min","cdhd_curr_cd_filled_idx-max","cdhd_curr_cd_filled_idx-sum","cdhd_curr_cd_filled_idx-median","cdhd_curr_cd_filled_idx-var","cdhd_curr_cd_filled_idx-peak_to_peak","is_risk_trans_chnl_filled_idx-sum","is_highrisk_MCC-cnt_1","cur_tot_cnt-cnt_0","hist_query_cnt-sum","hist_query_cnt-cnt_0","hist_query_cnt-cnt_1","is_norm_rate-sum","is_norm_rate-cnt_0","is_norm_rate-cnt_1","day7_freq_cnt-cnt_0","is_HR_AIRC_cnt_x","is_HR_AIRC_2_cnt_x","more_trans_st_cnt","more_tcc_cnt","more_Rc_cnt","more_RScc_cnt","more_pemc_cnt","more_Pcc_cnt","more_OTS_cnt","is_HR_FSC_cnt_x","is_HR_FSCR_cnt_x","is_HR_CCC_cnt_x","is_HR_CCR_cnt_x","is_HR_card_media_cnt_x","is_HR_CAC_cnt_x","is_HR_AIRC_cnt_y","is_HR_Term_cnt_y","is_HR_AIRC_2_cnt_y","is_HR_TS_cnt","is_HR_Tccs_cnt","is_HR_Rc_cnt","is_HR_RScc_cnt","is_HR_RSCR_cnt","is_HR_pemc_cnt","is_HR_Pcc_cnt","is_HR_OTS_cnt","is_HR_FSC_cnt_y","is_HR_FSCR_cnt_y","is_HR_CCC_cnt_y","is_HR_CCR_cnt_y","is_HR_card_media_cnt_y","is_HR_CAC_cnt_y","not_HR_AIRC_cnt","not_HR_Term_cnt","not_HR_AIRC_2_cnt","not_HR_Term_2_cnt","not_HR_TS_cnt","not_HR_Tccs_cnt","not_HR_Rc_cnt","not_HR_RScc_cnt","not_HR_pemc_cnt","not_HR_Pcc_cnt","not_HR_OTS_cnt","not_HR_FSC_cnt","not_HR_FSCR_cnt","not_HR_CCC_cnt","not_HR_CCR_cnt","not_HR_card_media_cnt","not_HR_CAC_cnt","HR_AIRC_ratio","HR_AIRC_2_ratio","HR_TS_ratio","HR_Tccs_ratio","HR_Rc_ratio","HR_RScc_ratio","HR_pemc_ratio","HR_Pcc_ratio","HR_OTS_ratio","HR_FSC_ratio","HR_FSCR_ratio","HR_CCC_ratio","HR_CCR_ratio","HR_card_media_ratio","HR_CAC_ratio","trans_cnt","deposit_cards_cnt","credit_cards_cnt","credit_cards_ratio","deposit_trans_cnt","credit_trans_cnt","credit_trans_ratio","RMB_10_cnt","RMB_bk_5_cnt","RMB_bk_6_cnt","fenqi_cnt","chaxun_cnt","LR_county","is_NN","HR_hour_cd","LR_hour_mc","LR_weekday_mc","cz_mcc_cnt","sh_mcc_cnt","LR_RC","LR_RC_min","HR_chnl_MFI","LR_TCC_MFI","LR_TCC_PP","HR_TCC_max","HR_TCC_MFI","HR_AIRC_MFI","HR_CANL_MFI","LH_CAC_max","LR_CCR_max","HR_risk1","HR_risk2","HR_risk3","HR_risk5","HR_risk6","HR_risk7","LH_has_trans_month","LH_min_apply_delta","HR_month_med","HR_month_max","LR_fdc_mcc","LR_jt_mcc","LR_jy_mcc","LR_qc_mcc","LR_risk_mcc","LR_sy_mcc","LR_ws_mcc","LR_xx_mcc","LR_zb_mcc","LR_zf_mcc","LR_zs_mcc"], axis=1,inplace=False)

df_All_T = df_All[(df_All["label"] == 0) | (df_All["label"] == 1)]

df_All_0 = df_All_T[(df_All_T["label"]==0)]
df_All_1 = df_All_T[(df_All_T["label"]==1)]
df_All_1 = df_All_1.sample(frac=0.5)
df_All_train = pd.concat([df_All_0, df_All_1], axis=0)



df_All_test = df_All[(df_All["label"] != 0) & (df_All["label"] != 1)]


#####################################################
#####################################################
df_All_train = shuffle(df_All_train)
X_train = df_All_train.drop(["certid", "label"], axis=1, inplace=False)
y_train = df_All_train["label"]
clf = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, gamma=0.01, subsample=0.8, colsample_bytree=0.8,
                    objective='binary:logistic', reg_alpha=0.1, reg_lambda=0.1, seed=27)
clf = clf.fit(X_train, y_train)
X_test = df_All_test.drop(["certid", "label"], axis=1, inplace=False)
pred = clf.predict(X_test).T
cerid_arr = np.array(df_All_test["certid"]).T
cerid_arr = np.vstack((cerid_arr, pred))


for i in range(0):
    savename = "temp_nomchnt_" + str(i) + ".csv"
    print savename
    df_All_train = shuffle(df_All_train)
    X_train = df_All_train.drop(["certid", "label"], axis=1, inplace=False)
    y_train = df_All_train["label"]
    clf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,gamma=0.01,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_alpha=0.1, reg_lambda=0.1,seed=27)
    clf = clf.fit(X_train, y_train)
    X_test = df_All_test.drop(["certid", "label"], axis=1, inplace=False)
    pred = clf.predict(X_test).T
    cerid_arr = np.vstack((cerid_arr,pred))
    np.savetxt(savename, cerid_arr.T, delimiter=',', fmt="%s")

np.savetxt("test2.csv",cerid_arr.T,delimiter=',', fmt = "%s")


