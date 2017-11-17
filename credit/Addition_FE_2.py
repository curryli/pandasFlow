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
import datetime
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

df_All = pd.read_csv("agg_cat.csv", sep=',')
#df_All = pd.read_csv("train_notest.csv", sep=',')
#df_All = pd.read_csv("train_1109_xyk.csv", sep=',')



# df_All_stat_0 = pd.read_csv("agg_cert.csv", sep=',')
# df_All = pd.merge(left=df_All, right=df_All_stat_0, how='left', left_on='certid', right_on='certid')

df_All_stat_0 = pd.read_csv("agg_math.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_0, how='left', left_on='certid', right_on='certid')

#df_All = df_All[(df_All["label"]==0) | (df_All["label"]==1)]

df_All_stat = pd.read_csv("translabel_stat.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat, how='left', left_on='certid', right_on='certid')

df_All_stat_2 = pd.read_csv("count_label.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_2, how='left', left_on='certid', right_on='certid')

df_All_stat_3 = pd.read_csv("count_label_isnot.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_3, how='left', left_on='certid', right_on='certid')

df_All_stat_4 = pd.read_csv("groupstat_2.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_4, how='left', left_on='certid', right_on='certid')

df_All_stat_5 = pd.read_csv("groupMCC.csv", sep=',')
df_All = pd.merge(left=df_All, right=df_All_stat_5, how='left', left_on='certid', right_on='certid')


############################################
############################################
############################################


def is_LR_RC(x):
    if(x >9):
        return 1
    else:
        return 0
df_All["LR_RC"] = df_All["resp_cd_filled_idx-countDistinct"].map(lambda x: is_LR_RC(x))


def is_LR_RC_min(x):
    if(x in [1,9,3]):
        return 1
    else:
        return 0
df_All["LR_RC_min"] = df_All["resp_cd_filled_idx-min"].map(lambda x: is_LR_RC_min(x))

def is_LR_RC_MFI(x):
    if(x in [1,2]):
        return 1
    else:
        return 0
df_All["LR_RC_MFI"] = df_All["resp_cd_filled_idx-most_frequent_item"].map(lambda x: is_LR_RC_MFI(x))

def is_LH_TC_CD(x):
    if(x <10):
        return 2
    elif (x>200):
        return 1
    else:
        return 0
df_All["LH_TC_CD"] = df_All["term_cd_filled_idx-countDistinct"].map(lambda x: is_LH_TC_CD(x))

def is_HR_TC_MFI(x):
    if(x ==2):
        return 1
    else:
        return 0
df_All["HR_TC_MFI"] = df_All["term_cd_filled_idx-most_frequent_item"].map(lambda x: is_HR_TC_MFI(x))


def is_HR_chnl_MFI(x):
    if(x ==2):
        return 1
    else:
        return 0
df_All["HR_chnl_MFI"] = df_All["term_cd_filled_idx-most_frequent_item"].map(lambda x: is_HR_chnl_MFI(x))


def is_LR_TCC_CD(x):
    if(x >4):
        return 1
    else:
        return 0
df_All["LR_TCC_CD"] = df_All["trans_curr_cd_filled_idx-countDistinct"].map(lambda x: is_LR_TCC_CD(x))


def is_LR_TCC_MFI(x):
    if(x <8):
        return 1
    else:
        return 0
df_All["LR_TCC_MFI"] = df_All["trans_curr_cd_filled_idx-most_frequent_cnt"].map(lambda x: is_LR_TCC_MFI(x))


def is_LR_TCC_PP(x):
    if(x >12):
        return 1
    else:
        return 0
df_All["LR_TCC_PP"] = df_All["trans_curr_cd_filled_idx-peak_to_peak"].map(lambda x: is_LR_TCC_PP(x))

def is_HR_TCC_max(x):
    if(x in [0,1,2]):
        return 1
    else:
        return 0
df_All["HR_TCC_max"] = df_All["trans_id_cd_filled_idx-max"].map(lambda x: is_HR_TCC_max(x))


def is_HR_TCC_MFI(x):
    if(x <8):
        return 1
    else:
        return 0
df_All["HR_TCC_MFI"] = df_All["trans_id_cd_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_TCC_MFI(x))

def is_HR_TS_MFC(x):
    if(x <8):
        return 1
    else:
        return 0
df_All["HR_TS_MFC"] = df_All["trans_st_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_TS_MFC(x))

def is_LR_TS_PP(x):
    if(x <8):
        return 1
    else:
        return 0
df_All["LR_TS_PP"] = df_All["trans_st_filled_idx-peak_to_peak"].map(lambda x: is_LR_TS_PP(x))

def is_HR_AIRC_MFC(x):
    if(x <8):
        return 1
    else:
        return 0
df_All["HR_AIRC_MFC"] = df_All["auth_id_resp_cd_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_AIRC_MFC(x))


def is_HR_AIRC_MFI(x):
    if(x in [0,1]):
        return 1
    else:
        return 0
df_All["HR_AIRC_MFI"] = df_All["auth_id_resp_cd_filled_idx-most_frequent_item"].map(lambda x: is_HR_AIRC_MFI(x))

def is_HR_CANL_CD(x):
    if(x <10):
        return 1
    else:
        return 0
df_All["HR_CANL_CD"] = df_All["card_accprt_nm_loc_filled_idx-countDistinct"].map(lambda x: is_HR_CANL_CD(x))


def is_HR_CANL_MFC(x):
    if(x <4):
        return 1
    else:
        return 0
df_All["HR_CANL_MFC"] = df_All["card_accprt_nm_loc_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_CANL_MFC(x))


def is_HR_CANL_MFI(x):
    if(x in [3,4,5]):
        return 1
    else:
        return 0
df_All["HR_CANL_MFI"] = df_All["card_accprt_nm_loc_filled_idx-most_frequent_item"].map(lambda x: is_HR_CANL_MFI(x))


def is_LH_CAC_CD(x):
    if(x >6):
        return 2
    elif (x ==1):
        return 1
    else:
        return 0
df_All["LH_CAC_CD"] = df_All["card_attr_cd_filled_idx-countDistinct"].map(lambda x: is_LH_CAC_CD(x))


def is_HR_CAC_max(x):
    if(x ==0):
        return 1
    else:
        return 0
df_All["LH_CAC_max"] = df_All["card_attr_cd_filled_idx-max"].map(lambda x: is_HR_CAC_max(x))

def is_HR_CAC_MFC(x):
    if(x <13):
        return 1
    else:
        return 0
df_All["LH_CAC_MFC"] = df_All["card_attr_cd_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_CAC_MFC(x))


def is_LR_CCR_max(x):
    if(x >3):
        return 1
    else:
        return 0
df_All["LR_CCR_max"] = df_All["cdhd_conv_rt_filled_idx-max"].map(lambda x: is_LR_CCR_max(x))


def is_HR_CCR_MFC(x):
    if(x <12):
        return 1
    else:
        return 0
df_All["HR_CCR_MFC"] = df_All["cdhd_conv_rt_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_CCR_MFC(x))

def is_LR_CCC_CD(x):
    if(x >6):
        return 1
    else:
        return 0
df_All["LR_CCC_CD"] = df_All["cdhd_curr_cd_filled_idx-countDistinct"].map(lambda x: is_LR_CCC_CD(x))

def is_HR_CCC_MFC(x):
    if(x <12):
        return 1
    else:
        return 0
df_All["HR_CCC_MFC"] = df_All["cdhd_curr_cd_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_CCC_MFC(x))


def is_HR_FCR_MFC(x):
    if(x <8):
        return 1
    else:
        return 0
df_All["HR_FCR_MFC"] = df_All["fwd_settle_conv_rt_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_FCR_MFC(x))


def is_LR_FCC_CD(x):
    if(x >6):
        return 1
    else:
        return 0
df_All["LR_FCC_CD"] = df_All["fwd_settle_cruu_cd_filled_idx-countDistinct"].map(lambda x: is_LR_FCC_CD(x))

def is_HR_FCC_MFC(x):
    if(x <10):
        return 1
    else:
        return 0
df_All["HR_FCC_MFC"] = df_All["fwd_settle_cruu_cd_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_FCC_MFC(x))


def is_HR_risk1(x):
    if(x >14):
        return 1
    else:
        return 0
df_All["HR_risk1"] = df_All["is_risk_card_accprt_nm_loc_filled_idx-sum"].map(lambda x: is_HR_risk1(x))


def is_HR_risk2(x):
    if(x >0):
        return 1
    else:
        return 0
df_All["HR_risk2"] = df_All["is_risk_card_attr_cd_filled_idx-sum"].map(lambda x: is_HR_risk2(x))


def is_HR_risk3(x):
    if(x >0):
        return 1
    else:
        return 0
df_All["HR_risk3"] = df_All["is_risk_fwd_settle_conv_rt_filled_idx-sum"].map(lambda x: is_HR_risk3(x))


def is_HR_risk4(x):
    if(x >0):
        return 1
    else:
        return 0
df_All["HR_risk4"] = df_All["is_risk_iss_ins_cd_filled_idx-sum"].map(lambda x: is_HR_risk4(x))

def is_HR_risk5(x):
    if(x >7):
        return 1
    else:
        return 0
df_All["HR_risk5"] = df_All["is_risk_mchnt_cd_filled_idx-sum"].map(lambda x: is_HR_risk5(x))


def is_HR_risk6(x):
    if(x >0):
        return 1
    else:
        return 0
df_All["HR_risk6"] = df_All["is_risk_rcv_settle_conv_rt_filled_idx-sum"].map(lambda x: is_HR_risk6(x))


def is_HR_risk7(x):
    if(x >0):
        return 1
    else:
        return 0
df_All["HR_risk7"] = df_All["is_risk_resp_cd_filled_idx-sum"].map(lambda x: is_HR_risk7(x))

def is_HR_risk8(x):
    if(x >14):
        return 1
    else:
        return 0
df_All["HR_risk8"] = df_All["is_risk_term_cd_filled_idx-sum"].map(lambda x: is_HR_risk8(x))


def is_HR_IIC_MFC(x):
    if(x <8):
        return 1
    else:
        return 0
df_All["HR_IIC_MFC"] = df_All["iss_ins_cd_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_IIC_MFC(x))


def is_HR_MC_CD(x):
    if(x <5):
        return 1
    else:
        return 0
df_All["HR_MC_CD"] = df_All["mcc_cd_filled_idx-countDistinct"].map(lambda x: is_HR_MC_CD(x))


def is_HR_MC_MFC(x):
    if(x <4):
        return 1
    else:
        return 0
df_All["HR_MC_MFC"] = df_All["mcc_cd_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_MC_MFC(x))


def is_HR_MCF_CD(x):
    if(x <9):
        return 1
    else:
        return 0
df_All["HR_MCF_CD"] = df_All["mchnt_cd_filled_idx-countDistinct"].map(lambda x: is_HR_MCF_CD(x))


def is_HR_mchnt_MFC(x):
    if(x in [1,2]):
        return 1
    else:
        return 0
df_All["HR_mchnt_MFC"] = df_All["mchnt_cd_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_mchnt_MFC(x))

def is_HR_OTS_MFC(x):
    if(x <10):
        return 1
    else:
        return 0
df_All["HR_OTS_MFC"] = df_All["orig_trans_st_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_OTS_MFC(x))

def is_LR_PCC_CD(x):
    if(x >12):
        return 1
    else:
        return 0
df_All["LR_PCC_CD"] = df_All["pos_cond_cd_filled_idx-countDistinct"].map(lambda x: is_LR_PCC_CD(x))

def is_HR_PCC_MFC(x):
    if(x <8):
        return 1
    else:
        return 0
df_All["HR_PCC_MFC"] = df_All["pos_cond_cd_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_PCC_MFC(x))

def is_HR_RSC_MFC(x):
    if(x <8):
        return 1
    else:
        return 0
df_All["HR_RSC_MFC"] = df_All["rcv_settle_conv_rt_filled_idx-most_frequent_cnt"].map(lambda x: is_HR_RSC_MFC(x))



def is_LR_RSCC_CD(x):
    if(x >6):
        return 1
    else:
        return 0
df_All["LR_RSCC_CD"] = df_All["rcv_settle_curr_cd_filled_idx-countDistinct"].map(lambda x: is_LR_RSCC_CD(x))

def is_LR_fdc_mcc(x):
    if(x >3):
        return 1
    else:
        return 0
df_All["LR_fdc_mcc"] = df_All["fdc_mcc_cnt"].map(lambda x: is_LR_fdc_mcc(x))


def is_HR_gg_mcc(x):
    if(x ==3):
        return 1
    else:
        return 0
df_All["HR_gg_mcc"] = df_All["gg_mcc_cnt"].map(lambda x: is_HR_gg_mcc(x))

def is_LR_jt_mcc(x):
    if(x >85):
        return 1
    else:
        return 0
df_All["LR_jt_mcc"] = df_All["jt_mcc_cnt"].map(lambda x: is_LR_jt_mcc(x))


def is_LR_jy_mcc(x):
    if(x >6):
        return 1
    else:
        return 0
df_All["LR_jy_mcc"] = df_All["jy_mcc_cnt"].map(lambda x: is_LR_jy_mcc(x))

def is_LR_qc_mcc(x):
    if(x >40):
        return 1
    else:
        return 0
df_All["LR_qc_mcc"] = df_All["qc_mcc_cnt"].map(lambda x: is_LR_qc_mcc(x))

def is_LR_risk_mcc(x):
    if(x >12):
        return 1
    else:
        return 0
df_All["LR_risk_mcc"] = df_All["risk_mcc_cnt"].map(lambda x: is_LR_risk_mcc(x))


def is_LR_sy_mcc(x):
    if(x >20):
        return 1
    else:
        return 0
df_All["LR_sy_mcc"] = df_All["sy_mcc_cnt"].map(lambda x: is_LR_sy_mcc(x))


def is_LR_ws_mcc(x):
    if(x >15):
        return 1
    else:
        return 0
df_All["LR_ws_mcc"] = df_All["ws_mcc_cnt"].map(lambda x: is_LR_ws_mcc(x))


def is_LR_xx_mcc(x):
    if(x >20):
        return 1
    else:
        return 0
df_All["LR_xx_mcc"] = df_All["xx_mcc_cnt"].map(lambda x: is_LR_xx_mcc(x))


def is_LR_zb_mcc(x):
    if(x >10):
        return 1
    else:
        return 0
df_All["LR_zb_mcc"] = df_All["zb_mcc_cnt"].map(lambda x: is_LR_zb_mcc(x))

def is_LR_zf_mcc(x):
    if(x >50):
        return 1
    else:
        return 0
df_All["LR_zf_mcc"] = df_All["zf_mcc_cnt"].map(lambda x: is_LR_zf_mcc(x))


def is_LR_zs_mcc(x):
    if(x >20):
        return 1
    else:
        return 0
df_All["LR_zs_mcc"] = df_All["zs_mcc_cnt"].map(lambda x: is_LR_zs_mcc(x))


def is_LH_has_trans_month(x):
    if(x >14):
        return 2
    elif (x<5):
        return 1
    else:
        return 0
df_All["LH_has_trans_month"] = df_All["has_trans_month"].map(lambda x: is_LH_has_trans_month(x))


def is_LH_min_apply_delta(x):
    if(x >400):
        return 2
    elif (x<100):
        return 1
    else:
        return 0
df_All["LH_min_apply_delta"] = df_All["min_apply_delta"].map(lambda x: is_LH_min_apply_delta(x))


def is_LR_m3(x):
    if(x >80):
        return 1
    else:
        return 0
df_All["LR_m3"] = df_All["month_cnt_3"].map(lambda x: is_LR_m3(x))


def is_HR_month_CD(x):
    if(x in [1,2,3,4]):
        return 1
    else:
        return 0
df_All["HR_month_CD"] = df_All["month-countDistinct"].map(lambda x: is_HR_month_CD(x))


def is_HR_month_med(x):
    if(x <7):
        return 1
    else:
        return 0
df_All["HR_month_med"] = df_All["month-median"].map(lambda x: is_HR_month_med(x))


def is_HR_month_MFC(x):
    if(x in [4,5,6,7]):
        return 1
    else:
        return 0
df_All["HR_month_MFC"] = df_All["month-most_frequent_item"].map(lambda x: is_HR_month_MFC(x))


def is_HR_month_PP(x):
    if(x <5):
        return 1
    else:
        return 0
df_All["HR_month_PP"] = df_All["month-peak_to_peak"].map(lambda x: is_HR_month_PP(x))


def is_HR_month_max(x):
    if(x in [1,2,3]):
        return 1
    else:
        return 0
df_All["HR_month_max"] = df_All["trans_month_max"].map(lambda x: is_HR_month_max(x))


def is_LR_WK_med(x):
    if(x in [2,3,4]):
        return 1
    else:
        return 0
df_All["LR_WK_med"] = df_All["weekday-median"].map(lambda x: is_LR_WK_med(x))


def is_LR_WK_MFC(x):
    if(x in [1,2,3,4]):
        return 1
    else:
        return 0
df_All["LR_WK_MFC"] = df_All["weekday-most_frequent_cnt"].map(lambda x: is_LR_WK_MFC(x))



df_All = df_All.fillna(-1)

df_All = shuffle(df_All)

save_lst = ["certid","LR_RC","LR_RC_min","LR_RC_MFI","LH_TC_CD","HR_TC_MFI","HR_chnl_MFI","LR_TCC_CD","LR_TCC_MFI","LR_TCC_PP","HR_TCC_max","HR_TCC_MFI","HR_TS_MFC","LR_TS_PP","HR_AIRC_MFC","HR_AIRC_MFI","HR_CANL_CD","HR_CANL_MFC","HR_CANL_MFI","LH_CAC_CD","LH_CAC_max","LH_CAC_MFC","LR_CCR_max","HR_CCR_MFC","LR_CCC_CD","HR_CCC_MFC","HR_FCR_MFC","LR_FCC_CD","HR_FCC_MFC","HR_risk1","HR_risk2","HR_risk3","HR_risk4","HR_risk5","HR_risk6","HR_risk7","HR_risk8","HR_IIC_MFC","HR_MC_CD","HR_MC_MFC","HR_MCF_CD","HR_mchnt_MFC","HR_OTS_MFC","LR_PCC_CD","HR_PCC_MFC","HR_RSC_MFC","LR_RSCC_CD","LR_fdc_mcc","HR_gg_mcc","LR_jt_mcc","LR_jy_mcc","LR_qc_mcc","LR_risk_mcc","LR_sy_mcc","LR_ws_mcc","LR_xx_mcc","LR_zb_mcc","LR_zf_mcc","LR_zs_mcc","LH_has_trans_month","LH_min_apply_delta","LR_m3","HR_month_CD","HR_month_med","HR_month_MFC","HR_month_PP","HR_month_max","LR_WK_med","LR_WK_MFC"]
df_All[save_lst].to_csv("addition_stat_2.csv",index=False)
##########################################################################


# df_X = df_All.drop( ["certid","label"], axis=1,inplace=False)
# print df_X.columns
#
# df_y = df_All["label"]
#
# X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)
# X_cols = X_train.columns
# sc = StandardScaler()    #MinMaxScaler()    不好
#
# #print X_train.loc[:1]
#
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# #clf = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,gamma=0.05,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_lambda=1,seed=27)
# clf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,gamma=0.01,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic', reg_alpha=0.1, reg_lambda=0.1,seed=27)
#
# clf.fit(X_train, y_train)
#
# pred = clf.predict(X_test)
#
# cm1=confusion_matrix(y_test,pred)
# print  cm1
#
# precision_p = float(cm1[0][0])/float((cm1[0][0] + cm1[1][0]))
# recall_p = float(cm1[0][0])/float((cm1[0][0] + cm1[0][1]))
# F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)
#
# print ("Precision:", precision_p)
# print ("Recall:", recall_p)
# print ("F1_Score:", F1_Score)
#
# FE_ip_tuples = zip(X_cols, clf.feature_importances_)
# pd.DataFrame(FE_ip_tuples).to_csv("FE_IP_xgboost_add.csv",index=True)
#
#
# #Compute precision, recall, F-measure and support for each class
# # print "weighted\n"
# # print precision_recall_fscore_support(y_test,pred, average='weighted')
#
# print "Each class\n"
# result = precision_recall_fscore_support(y_test,pred)
# #print result
# precision_0 = result[0][0]
# recall_0 = result[1][0]
# f1_0 = result[2][0]
# precision_1 = result[0][1]
# recall_1 = result[1][1]
# f1_1 = result[2][1]
# print "precision_0: ", precision_0,"  recall_0: ", recall_0, "  f1_0: ", f1_0

