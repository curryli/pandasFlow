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
df_All_stat_9 = pd.read_csv("mchnt_ana.csv", sep=',')
df_All_stat_9 = df_All_stat_9[["certid","has_mchnt_risk","has_mchntcd_risk"]]
df_All = pd.merge(left=df_All, right=df_All_stat_9, how='left', left_on='certid', right_on='certid')
#########################

label_df = pd.read_csv("train_label_encrypt.csv", sep=",", low_memory=False, error_bad_lines=False)
df_All = pd.merge(left=df_All, right=label_df, how='left', left_on='certid', right_on='certid')

#df_All =  df_All.drop(["month_No-cnt_16","month_No-cnt_17","month_No-cnt_18","month_No-cnt_19","month_No-cnt_20","fund_shortage-sum","cdhd_at-min","cdhd_at-max","cdhd_at-mean","cdhd_at-sum","cdhd_at-median","cdhd_at-var","Trans_at-count","stageInMonth-countDistinct","stageInMonth-min","stageInMonth-peak_to_peak","month_sum_17","month_sum_18","month_sum_19","month_cnt_14","month_cnt_15","month_cnt_16","month_cnt_17","month_cnt_18","month_cnt_19","mean_money_m15","mean_money_m16","mean_money_m17","mean_money_m18","mean_money_m19","trans_month_mean","card_attr_cd_filled_idx-most_frequent_item","card_attr_cd_filled_idx-min","card_attr_cd_filled_idx-median","card_attr_cd_filled_idx-peak_to_peak","is_risk_cdhd_curr_cd_filled_idx-sum","is_risk_fwd_settle_cruu_cd_filled_idx-sum","trans_st_filled_idx-countDistinct","trans_st_filled_idx-most_frequent_item","trans_st_filled_idx-min","trans_st_filled_idx-max","trans_st_filled_idx-sum","trans_st_filled_idx-median","is_risk_orig_trans_st_filled_idx-sum","is_risk_cdhd_conv_rt_filled_idx-sum","fwd_settle_cruu_cd_filled_idx-countDistinct","fwd_settle_cruu_cd_filled_idx-most_frequent_item","fwd_settle_cruu_cd_filled_idx-most_frequent_cnt","fwd_settle_cruu_cd_filled_idx-min","fwd_settle_cruu_cd_filled_idx-max","fwd_settle_cruu_cd_filled_idx-mean","fwd_settle_cruu_cd_filled_idx-sum","fwd_settle_cruu_cd_filled_idx-median","fwd_settle_cruu_cd_filled_idx-std","fwd_settle_cruu_cd_filled_idx-var","fwd_settle_cruu_cd_filled_idx-peak_to_peak","trans_curr_cd_filled_idx-countDistinct","trans_curr_cd_filled_idx-most_frequent_item","trans_curr_cd_filled_idx-most_frequent_cnt","trans_curr_cd_filled_idx-min","trans_curr_cd_filled_idx-max","trans_curr_cd_filled_idx-mean","trans_curr_cd_filled_idx-median","trans_curr_cd_filled_idx-std","trans_curr_cd_filled_idx-var","trans_curr_cd_filled_idx-peak_to_peak","is_risk_trans_st_filled_idx-sum","rcv_settle_curr_cd_filled_idx-countDistinct","rcv_settle_curr_cd_filled_idx-most_frequent_item","rcv_settle_curr_cd_filled_idx-most_frequent_cnt","rcv_settle_curr_cd_filled_idx-min","rcv_settle_curr_cd_filled_idx-max","rcv_settle_curr_cd_filled_idx-mean","rcv_settle_curr_cd_filled_idx-sum","rcv_settle_curr_cd_filled_idx-median","rcv_settle_curr_cd_filled_idx-std","rcv_settle_curr_cd_filled_idx-var","rcv_settle_curr_cd_filled_idx-peak_to_peak","is_risk_mcc_cd_filled_idx-sum","cdhd_conv_rt_filled_idx-countDistinct","cdhd_conv_rt_filled_idx-most_frequent_item","cdhd_conv_rt_filled_idx-most_frequent_cnt","cdhd_conv_rt_filled_idx-min","cdhd_conv_rt_filled_idx-max","cdhd_conv_rt_filled_idx-mean","cdhd_conv_rt_filled_idx-sum","cdhd_conv_rt_filled_idx-median","cdhd_conv_rt_filled_idx-std","cdhd_conv_rt_filled_idx-var","cdhd_conv_rt_filled_idx-peak_to_peak","is_risk_trans_curr_cd_filled_idx-sum","is_risk_card_media_cd_filled_idx-sum","is_risk_trans_id_cd_filled_idx-sum","pos_cond_cd_filled_idx-min","pos_cond_cd_filled_idx-peak_to_peak","fwd_settle_conv_rt_filled_idx-most_frequent_item","fwd_settle_conv_rt_filled_idx-most_frequent_cnt","fwd_settle_conv_rt_filled_idx-min","fwd_settle_conv_rt_filled_idx-max","fwd_settle_conv_rt_filled_idx-mean","fwd_settle_conv_rt_filled_idx-median","fwd_settle_conv_rt_filled_idx-var","fwd_settle_conv_rt_filled_idx-peak_to_peak","auth_id_resp_cd_filled_idx-most_frequent_item","resp_cd_filled_idx-min","orig_trans_st_filled_idx-countDistinct","orig_trans_st_filled_idx-most_frequent_item","orig_trans_st_filled_idx-most_frequent_cnt","orig_trans_st_filled_idx-min","orig_trans_st_filled_idx-sum","orig_trans_st_filled_idx-median","orig_trans_st_filled_idx-var","orig_trans_st_filled_idx-peak_to_peak","rcv_settle_conv_rt_filled_idx-countDistinct","rcv_settle_conv_rt_filled_idx-most_frequent_item","rcv_settle_conv_rt_filled_idx-most_frequent_cnt","rcv_settle_conv_rt_filled_idx-min","rcv_settle_conv_rt_filled_idx-max","rcv_settle_conv_rt_filled_idx-median","rcv_settle_conv_rt_filled_idx-peak_to_peak","is_risk_resp_cd_filled_idx-sum","is_risk_card_attr_cd_filled_idx-sum","is_risk_rcv_settle_curr_cd_filled_idx-sum","is_risk_pos_entry_md_cd_filled_idx-sum","cdhd_curr_cd_filled_idx-countDistinct","cdhd_curr_cd_filled_idx-most_frequent_item","cdhd_curr_cd_filled_idx-most_frequent_cnt","cdhd_curr_cd_filled_idx-min","cdhd_curr_cd_filled_idx-max","cdhd_curr_cd_filled_idx-sum","cdhd_curr_cd_filled_idx-median","cdhd_curr_cd_filled_idx-var","cdhd_curr_cd_filled_idx-peak_to_peak","is_risk_trans_chnl_filled_idx-sum","cur_tot_cnt-cnt_0","hist_query_cnt-cnt_1","is_norm_rate-sum","is_norm_rate-cnt_0","is_norm_rate-cnt_1","day7_freq_cnt-cnt_0","1hour_highrisk_MCC_cnt-cnt_0","is_HR_AIRC_cnt_x","is_HR_AIRC_2_cnt_x","more_trans_st_cnt","more_tcc_cnt","more_Rc_cnt","more_RScc_cnt","more_pemc_cnt","more_Pcc_cnt","more_OTS_cnt","is_HR_FSC_cnt_x","is_HR_FSCR_cnt_x","is_HR_CCC_cnt_x","is_HR_CCR_cnt_x","is_HR_card_media_cnt_x","is_HR_CAC_cnt_x","is_HR_AIRC_cnt_y","is_HR_Term_cnt_y","is_HR_AIRC_2_cnt_y","is_HR_TS_cnt","is_HR_Tccs_cnt","is_HR_Rc_cnt","is_HR_RScc_cnt","is_HR_RSCR_cnt","is_HR_pemc_cnt","is_HR_Pcc_cnt","is_HR_OTS_cnt","is_HR_FSC_cnt_y","is_HR_FSCR_cnt_y","is_HR_CCC_cnt_y","is_HR_CCR_cnt_y","is_HR_card_media_cnt_y","is_HR_CAC_cnt_y","not_HR_AIRC_cnt","not_HR_Term_cnt","not_HR_AIRC_2_cnt","not_HR_MC_2_cnt","not_HR_TS_cnt","not_HR_Tccs_cnt","not_HR_Rc_cnt","not_HR_RScc_cnt","not_HR_pemc_cnt","not_HR_Pcc_cnt","not_HR_OTS_cnt","not_HR_FSC_cnt","not_HR_FSCR_cnt","not_HR_CCC_cnt","not_HR_CCR_cnt","not_HR_card_media_cnt","not_HR_CAC_cnt","HR_AIRC_ratio","HR_Term_ratio","HR_AIRC_2_ratio","HR_TS_ratio","HR_Tccs_ratio","HR_Rc_ratio","HR_RScc_ratio","HR_RSCR_ratio","HR_pemc_ratio","HR_Pcc_ratio","HR_OTS_ratio","HR_FSC_ratio","HR_FSCR_ratio","HR_CCC_ratio","HR_CCR_ratio","HR_card_media_ratio","HR_CAC_ratio","trans_cnt","deposit_cards_cnt","credit_cards_ratio","deposit_trans_cnt","credit_trans_cnt","credit_trans_ratio","RMB_10_cnt","RMB_bk_5_cnt","RMB_bk_6_cnt","fenqi_cnt","chaxun_cnt","LR_county","is_NN","LR_hour_mc","LR_weekday_mc","fdc_mcc_cnt","cz_mcc_cnt","wl_mcc_cnt","sh_mcc_cnt","LR_RC","LR_RC_min","HR_chnl_MFI","LR_TCC_MFI","LR_TCC_PP","HR_TCC_MFI","HR_AIRC_MFI","HR_CANL_MFI","LH_CAC_max","LR_CCR_max","HR_risk1","HR_risk2","HR_risk5","HR_risk6","HR_risk7","LH_has_trans_month","HR_month_med","HR_month_PP","HR_month_max","LR_fdc_mcc","LR_jt_mcc","LR_jy_mcc","LR_qc_mcc","LR_risk_mcc","LR_sy_mcc","LR_ws_mcc","LR_xx_mcc","LR_zb_mcc","LR_zf_mcc","LR_zs_mcc"] , axis=1, inplace=False)

df_All = df_All.fillna(-1)

df_All_train = df_All[(df_All["label"] == 0) | (df_All["label"] == 1)]
df_All_test = df_All[(df_All["label"] != 0) & (df_All["label"] != 1)]

print df_All_train.shape
print df_All_test.shape




df_All_train = shuffle(df_All_train)

X_train = df_All_train.drop(["certid", "label"], axis=1, inplace=False)

y_train = df_All_train["label"]

#clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)
clf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,gamma=0.01,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_alpha=0.1, reg_lambda=0.1,seed=27)

# clf = GradientBoostingClassifier(random_state=100, n_estimators=100)
clf = clf.fit(X_train, y_train)


# joblib.dump(clf, "xgboost.mdl")
#
# clf = joblib.load("xgboost.mdl")
# print "model loaded sucessfully."


X_test = df_All_test.drop(["certid", "label"], axis=1, inplace=False)

pred = clf.predict(X_test).T

print pred.shape

cerid_arr = np.array(df_All_test["certid"]).T

result = np.vstack((cerid_arr,pred))
print np.savetxt("test.csv",result.T,delimiter=',', fmt = "%s")

