# -*- coding: utf-8 -*-
'''''This script demonstrates how to build a variational autoencoder with Keras. 
https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py 
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114 
'''  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import norm  
  
from keras.layers import Input, Dense, Lambda  
from keras.models import Model  
from keras import backend as K  
from keras import objectives  
 
 
from keras import regularizers

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#from keras.datasets import mnist  
#from keras.utils.visualize_util import plot  
import sys  

import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
 
print "start"
################################################################
df_All = pd.read_csv("idx_new_08_del.csv", sep=',') 
df_All = shuffle(df_All) 
 
df_dummies =  df_All[["settle_tp","settle_cycle","block_id","trans_fwd_st","sms_dms_conv_in","cross_dist_in","tfr_in_in","trans_md","source_region_cd","dest_region_cd","cups_card_in","cups_sig_card_in","card_class","card_attr","acq_ins_tp","fwd_ins_tp","rcv_ins_tp","iss_ins_tp","acpt_ins_tp","resp_cd1","resp_cd3","cu_trans_st","sti_takeout_in","trans_id","trans_tp","trans_chnl","card_media","trans_curr_cd","conn_md","msg_tp","msg_tp_conv","trans_proc_cd","trans_proc_cd_conv","mchnt_tp","pos_entry_md_cd","pos_cond_cd","pos_cond_cd_conv","term_tp","rsn_cd","addn_pos_inf","iss_ds_settle_in","acq_ds_settle_in","upd_in","pri_cycle_no","disc_in","fwd_settle_conv_rt","fwd_settle_curr_cd","rcv_settle_curr_cd","acq_ins_id_cd_BK","acq_ins_id_cd_RG","fwd_ins_id_cd_BK","fwd_ins_id_cd_RG","rcv_ins_id_cd_BK","rcv_ins_id_cd_RG","iss_ins_id_cd_BK","iss_ins_id_cd_RG","acpt_ins_id_cd_BK","acpt_ins_id_cd_RG","settle_fwd_ins_id_cd_BK","settle_fwd_ins_id_cd_RG","settle_rcv_ins_id_cd_BK","settle_rcv_ins_id_cd_RG"]]
 

df_X = pd.concat([df_All[["tfr_dt_tm","day_week","hour","trans_at","total_disc_at"]],df_dummies], axis=1).as_matrix()
#df_X = df_All[["hour","trans_at","total_disc_at"]]
vec_size = df_X.shape[1]

df_y = df_All["label"].as_matrix().reshape((-1,1))

x_train, x_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)

sc = MinMaxScaler()   #这里如果用StandardScaler，那么会出现负数，导致算出来的KL距离 loss会有负数，
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
 
##################################################################
 
#print x_train.shape, y_train.shape
samples_train = pd.DataFrame(np.concatenate((x_train, y_train),axis=1))           
Fraud_train = samples_train[samples_train.icol(-1) == 1]
Normal_train = samples_train[samples_train.icol(-1) == 0]

print "samples_train size: ", samples_train.shape, "Fraud_train size: ",Fraud_train.shape, "Normal_train size: ", Normal_train.shape 

X_fraud_T = Fraud_train.iloc[:,:-1].as_matrix()   #不包含最后一列（-1）
 
xlen = Fraud_train.shape[0]

x_fraud_train = X_fraud_T[:4*xlen/5]

x_fraud_test = X_fraud_T[4*xlen/5:]
 
#print "x_fraud_train size: ",x_fraud_train.shape, "x_fraud_test size: ", x_fraud_test.shape 


##################################################################



batch_size = 100
original_dim = vec_size
latent_dim = 2  
intermediate_dim = 64  
nb_epoch = 100 
epsilon_std = 1.0  
  
#my tips:encoding  
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
  
#my tips:Gauss sampling,sample Z  
def sampling(args):   
    z_mean, z_log_var = args  
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,  
                              stddev=epsilon_std)  
    return z_mean + K.exp(z_log_var / 2) * epsilon  
  
# note that "output_shape" isn't necessary with the TensorFlow backend  
# my tips:get sample z(encoded)  
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])   #图中z
  
# we instantiate these layers separately so as to reuse them later  
decoder_h = Dense(intermediate_dim, activation='relu')  
decoder_mean = Dense(original_dim, activation='sigmoid')  
h_decoded = decoder_h(z)   #图中f(z)
x_decoded_mean = decoder_mean(h_decoded)    #图中最后一个x
  
#my tips:loss(restruct X)+KL   目标函数有两项   
def vae_loss(x, x_decoded_mean):  
      #my tips:logloss   一项与自动编码器相同，要求从f(z)出来的样本重构原来的输入样本   由重构x与输入x均方差或逐点的交叉熵衡量
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)  
    #my tips:see paper's appendix B    另一项 要求经过Q(z|x)估计出的隐变量分布接近于标准正态分布    由衡量两个分布的相似度，当然是大名鼎鼎的KL距离。二是近似后验与真实后验的KL散度，至于KL散度为何简化成代码中的形式，看论文《Auto-Encoding Variational Bayes》中的附录B有证明。
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  
    return K.mean(xent_loss + kl_loss) 
  
vae = Model(x, x_decoded_mean)  
vae.compile(optimizer='rmsprop', loss=vae_loss)  
  


vae.fit(x_fraud_train, x_fraud_train,  
        shuffle=True,  
        epochs=nb_epoch,  
        verbose=2,   
        batch_size=batch_size,
        validation_data=(x_fraud_test, x_fraud_test))  
  
# build a model to project inputs on the latent space  
encoder = Model(x, z_mean)  
 
  
# build a digit generator that can sample from the learned distribution  使用vae  跟普通自动编码器不一样，我们这里只需要 剪掉编码器部分，直接把正态分布样本送入解码器即可
decoder_input = Input(shape=(latent_dim,))  
_h_decoded = decoder_h(decoder_input)  
_x_decoded_mean = decoder_mean(_h_decoded)  
generator = Model(decoder_input, _x_decoded_mean)  


X_vae = np.array([X_fraud_T[0]])
#print X_vae.shape


print "start generateing vae samples:"

for k in range(100000):
    if (k%1000 == 0):
        print k
    
    n = 15  # figure with 15x15 digits  
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian  
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian  
    #norm.pdf 正态分布的密度函数(根据x生成y)  norm.ppf正态分布中位数函数（根据y生成x）
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))  
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))  
      
    for i, yi in enumerate(grid_x):  
        for j, xi in enumerate(grid_y):  
            z_sample = np.array([[xi, yi]])  
            x_decoded = generator.predict(z_sample)  
            digit = x_decoded[0]
    X_vae = np.append(X_vae,digit)

X_vae = X_vae.reshape(-1,vec_size)
         
#print X_vae.shape

X_train_RF_fraud = np.concatenate((X_vae, X_fraud_T),axis=0)    
y_train_RF_fraud = np.ones(X_train_RF_fraud.shape[0]).reshape(-1,1)
   

 
Fraud_train_RF =  np.concatenate((X_train_RF_fraud, y_train_RF_fraud),axis=1)                           
  
All_train_RF =  np.concatenate((Fraud_train_RF, Normal_train.as_matrix()), axis = 0) 

All_train_RF = shuffle(All_train_RF)
x_train_RF =  All_train_RF[:,:-1]
y_train_RF =  All_train_RF[:,-1]

print "x_train_RF ", x_train_RF.shape, "x_test shape ", x_test.shape


#n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features="auto",max_leaf_nodes=None, bootstrap=True)

print  "start fit RF:\n"

clf = clf.fit(x_train_RF, y_train_RF)
  
print clf.score(x_test, y_test)

pred = clf.predict(x_test) 


#print type(y_test),type(pred)
#confusion_matrix=confusion_matrix(y_test.tolist(), pred.tolist())

#上面confusion_matrix要新起一个名字，否则会报错  TypeError: 'numpy.ndarray' object is not callable，和内部变量冲突了
confusion_matrix_2=confusion_matrix(y_test, pred)
print  "\nconfusion_matrix:\n"
print  confusion_matrix_2


precision_p = float(confusion_matrix_2[1][1])/float((confusion_matrix_2[0][1] + confusion_matrix_2[1][1]))
recall_p = float(confusion_matrix_2[1][1])/float((confusion_matrix_2[1][0] + confusion_matrix_2[1][1]))
 
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 
