# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 08:30:44 2017

@author: lixurui
"""

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
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
from sklearn.preprocessing import StandardScaler   
import tensorflow as tf #machine learning
import os #saving files
from datetime import datetime #logging
from sklearn.metrics import roc_auc_score as auc  #measuring accuracy

df_All = pd.read_csv("idx_new_08_del.csv", sep=',') 
df_All = shuffle(df_All) 
print df_All.shape[1]
 
#df_All.columns = ["pri_acct_no_conv","tfr_dt_tm","day_week","hour","trans_at","total_disc_at","settle_tp","settle_cycle","block_id","trans_fwd_st","trans_rcv_st","sms_dms_conv_in","cross_dist_in","tfr_in_in","trans_md","source_region_cd","dest_region_cd","cups_card_in","cups_sig_card_in","card_class","card_attr","acq_ins_tp","fwd_ins_tp","rcv_ins_tp","iss_ins_tp","acpt_ins_tp","resp_cd1","resp_cd2","resp_cd3","resp_cd4","cu_trans_st","sti_takeout_in","trans_id","trans_tp","trans_chnl","card_media","trans_id_conv","trans_curr_cd","conn_md","msg_tp","msg_tp_conv","trans_proc_cd","trans_proc_cd_conv","mchnt_tp","pos_entry_md_cd","pos_cond_cd","pos_cond_cd_conv","term_tp","rsn_cd","addn_pos_inf","iss_ds_settle_in","acq_ds_settle_in","upd_in","pri_cycle_no","disc_in","fwd_settle_conv_rt","rcv_settle_conv_rt","fwd_settle_curr_cd","rcv_settle_curr_cd","acq_ins_id_cd_BK","acq_ins_id_cd_RG","fwd_ins_id_cd_BK","fwd_ins_id_cd_RG","rcv_ins_id_cd_BK","rcv_ins_id_cd_RG","iss_ins_id_cd_BK","iss_ins_id_cd_RG","acpt_ins_id_cd_BK","acpt_ins_id_cd_RG","settle_fwd_ins_id_cd_BK","settle_fwd_ins_id_cd_RG","settle_rcv_ins_id_cd_BK","settle_rcv_ins_id_cd_RG","label"]#df_All = df_All.loc[:3000]
 
df_dummies =  df_All[["settle_tp","settle_cycle","block_id","trans_fwd_st","sms_dms_conv_in","cross_dist_in","tfr_in_in","trans_md","source_region_cd","dest_region_cd","cups_card_in","cups_sig_card_in","card_class","card_attr","acq_ins_tp","fwd_ins_tp","rcv_ins_tp","iss_ins_tp","acpt_ins_tp","resp_cd1","resp_cd3","cu_trans_st","sti_takeout_in","trans_id","trans_tp","trans_chnl","card_media","trans_curr_cd","conn_md","msg_tp","msg_tp_conv","trans_proc_cd","trans_proc_cd_conv","mchnt_tp","pos_entry_md_cd","pos_cond_cd","pos_cond_cd_conv","term_tp","rsn_cd","addn_pos_inf","iss_ds_settle_in","acq_ds_settle_in","upd_in","pri_cycle_no","disc_in","fwd_settle_conv_rt","fwd_settle_curr_cd","rcv_settle_curr_cd","acq_ins_id_cd_BK","acq_ins_id_cd_RG","fwd_ins_id_cd_BK","fwd_ins_id_cd_RG","rcv_ins_id_cd_BK","rcv_ins_id_cd_RG","iss_ins_id_cd_BK","iss_ins_id_cd_RG","acpt_ins_id_cd_BK","acpt_ins_id_cd_RG","settle_fwd_ins_id_cd_BK","settle_fwd_ins_id_cd_RG","settle_rcv_ins_id_cd_BK","settle_rcv_ins_id_cd_RG"]]

#print df_dummies.loc[:30]

#df_All = pd.concat([df_All[["pri_acct_no_conv_filled","day_week_filled","hour_filled","tfr_dt_tm_filled","trans_at_filled","total_disc_at_filled"]],df_dummies, df_All["label_filled"]], axis=1)



df_X = pd.concat([df_All[["tfr_dt_tm","day_week","hour","trans_at","total_disc_at"]],df_dummies], axis=1)#.as_matrix()
#df_X = df_All[["hour","trans_at","total_disc_at"]]
print df_X.shape

df_y = df_All["label"]#.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)
 

sc = StandardScaler()

#print X_train.loc[:1]

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

  # Parameters
learning_rate = 0.01 #how fast to learn? too low, too long to converge, too high overshoot minimum
training_epochs = 10
batch_size = 256
display_step = 1

# Network Parameters (neurons )
n_hidden_1 = 50 # 1st layer num features
#n_hidden_2 = 15 # 2nd layer num features
n_input = X_train.shape[1] # 67


X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    #'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    #'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    #'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_input])),
    #'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    #input times weight + bias. Activate! It rhymes.
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    return layer_1


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    return layer_1

# Construct model
#feed the input data into the encoder.
encoder_op = encoder(X)
#and learned representation to the decoder
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define batch mse
batch_mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2), 1)

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#this is gradient descent. Stochastic gradient descent. Backpropagate to update weights!
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# TRAIN StARTS
save_model = 'AE_TF_model_1.ckpt'
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    now = datetime.now()
    sess.run(init)
    total_batch = int(X_train.shape[0]/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            #pick a random datapoint from the batch
            batch_idx = np.random.choice(X_train.shape[0], batch_size)
            batch_xs = X_train[batch_idx]
            # Run optimization op (backprop) and cost op (to get loss value)
            #recursive weight updating via gradient computation. 
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            
        # Display logs per epoch step
        if epoch % display_step == 0:
            train_batch_mse = sess.run(batch_mse, feed_dict={X: X_train})
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c), 
                  "Train auc=", "{:.6f}".format(auc(y_train, train_batch_mse)), 
                  "Time elapsed=", "{}".format(datetime.now() - now))

    print("Optimization Finished!")
    
    save_path = saver.save(sess, save_model)
    print("Model saved in file: %s" % save_path)


load_model = 'AE_TF_model_1.ckpt.meta' 
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    now = datetime.now()
    
    #saver.restore(sess, load_model)
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    test_batch_mse = sess.run(batch_mse, feed_dict={X: X_test})
    
    print("Test auc score: {:.6f}".format(auc(y_test, test_batch_mse)))





##n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
#clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features="auto",max_leaf_nodes=None, bootstrap=True)
#
#clf = clf.fit(X_train, y_train)
#  
#print clf.score(X_test, y_test)
#
#pred = clf.predict(X_test) 
#
#confusion_matrix=confusion_matrix(y_test,pred)
#print  confusion_matrix
#
#
#precision_p = float(confusion_matrix[1][1])/float((confusion_matrix[0][1] + confusion_matrix[1][1]))
#recall_p = float(confusion_matrix[1][1])/float((confusion_matrix[1][0] + confusion_matrix[1][1]))
# 
#print ("Precision:", precision_p) 
#print ("Recall:", recall_p) 

