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
  


#http://blog.csdn.net/shine19930820/article/details/71713680


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


# 将训练集切分为两部分，一部分用于训练GBDT模型，另一部分输入到训练好的GBDT模型生成GBDT特征，然后作为LR的特征。这样分成两部分是为了防止过拟合。
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)


#n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
grd = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100,max_depth=10, min_samples_leaf =60, min_samples_split =1200, max_features=10, subsample=0.7, random_state=10)
  
grd_enc = OneHotEncoder()
# 调用LR分类模型。
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features="auto",max_leaf_nodes=None, bootstrap=True)

'''使用X_train训练GBDT模型，后面用此模型构造特征'''
grd.fit(X_train, y_train)


#apply函数return： [n_samples, n_estimators, n_classes]
# fit one-hot编码器
grd_enc.fit(grd.apply(X_train)[:, :, 0])
''' 
使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
'''
clf.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
# 用训练好的LR模型多X_test做预测

X_onehot = grd_enc.transform(grd.apply(X_test)[:, :, 0])
print "X_onehot shape: ", X_onehot.shape
 

pred2 = clf.predict(X_onehot)

print "y_test shape: ", y_test.shape
print "predict shape: ", pred2.shape


confusion_matrix_2=confusion_matrix(y_test,pred2)
print  confusion_matrix_2


precision_p2 = float(confusion_matrix_2[1][1])/float((confusion_matrix_2[0][1] + confusion_matrix_2[1][1]))
recall_p2 = float(confusion_matrix_2[1][1])/float((confusion_matrix_2[1][0] + confusion_matrix_2[1][1]))
 
print ("Precision:", precision_p2) 
print ("Recall:", recall_p2) 


# [[28999  1139]
# [ 1183  5835]]