# -*- coding: utf-8 -*-


import pandas as pd                #data analysing tool for python
import matplotlib.pyplot as plt    #data visualising tool for python
import numpy as np
np.random.seed(1234)

from IPython.display import display
from sklearn.preprocessing import StandardScaler,MinMaxScaler                   # for normalization of our data
 
from sklearn.model_selection import cross_val_score, GridSearchCV  #using Kfold and if needed, GridSearch object in analysis
from sklearn.utils import shuffle                                  # shuffling our own made dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import recall_score, precision_score
 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score as auc 
from sklearn.metrics import f1_score
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier 

import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
 

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



labelName="label" 
cardName = "pri_acct_no_conv" 
runEpoch=30
 
out_dim = 2
BS = 128
#runLoop = 50

#Alldata = pd.read_csv('convert_5_card.csv')
#Alldata = pd.read_csv('convert_5_card_GBDT.csv')
Alldata = pd.read_csv('convert5_weika_GBDT_07.csv')

Fraud = Alldata[Alldata.label == 1]
Normal = Alldata[Alldata.label == 0]
print "Ori Fraud shape:", Fraud.shape, "Ori Normal shape:", Normal.shape

card_c_F = Fraud['pri_acct_no_conv'].drop_duplicates()#涉欺诈交易卡号

#未出现在欺诈样本中的正样本数据
fine_N=Normal[(~Normal['pri_acct_no_conv'].isin(card_c_F))]
print "True Fraud shape:", Fraud.shape, "True Normal shape:", fine_N.shape

Alldata = pd.concat([Fraud,fine_N], axis = 0)


###############################
Alldata = shuffle(Alldata)

#Alldata = Alldata.head(2000)

train_all,test_all=train_test_split(Alldata, test_size=0.2)


#train_all = pd.read_csv('train-converted_4.csv')
#test_all = pd.read_csv('train-converted_4.csv')



y_train = train_all.label
y_test = test_all.label


#ex_dict = {0:[0,1],1:[1,0]}
 
#y_train = y_train.map(lambda x: ex_dict[x]) 
#y_test = y_test.map(lambda x: ex_dict[x])

y_train = np.array([1-y_train, y_train]).T
y_test = np.array([ 1 - y_test, y_test]).T

#print y_train.shape, y_test.shape

X_train = train_all.drop(labelName, axis = 1, inplace=False).drop(cardName, axis = 1, inplace=False)  
X_test = test_all.drop(labelName, axis = 1, inplace=False).drop(cardName, axis = 1, inplace=False)  

print X_train.columns

print len(X_train.columns)

size_data = X_train.shape[1];
print "size_data= ", size_data

timesteps = 5
data_dim = size_data/timesteps
print "data dimension=", data_dim


sc = StandardScaler()    #MinMaxScaler()    不好

#print X_train.loc[:1]

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

train_size = X_train.shape[0]  
 

global index_in_epoch
contents = X_train
labels = y_train


#https://stackoverflow.com/questions/41454511/tensorflow-how-is-dataset-train-next-batch-defined
def get_next_batch(batch_size, index_in_epoch, contents, labels):
    start = index_in_epoch
    index_in_epoch = index_in_epoch + batch_size
    end = index_in_epoch
    if index_in_epoch<train_size:
        return contents[start:end], labels[start:end]
    else:
        return contents[train_size-batch_size:train_size], labels[train_size-batch_size:train_size]

 
_X = tf.placeholder(tf.float32, shape=[None, timesteps, data_dim], name='input')
_y = tf.placeholder(tf.float32, [None, out_dim], name='label')
dropout = tf.placeholder(tf.float32, name='dropout')
batch_size = tf.placeholder(tf.int32, name='batch_size')  # 注意类型必须为 tf.int32
 
num_units = 300
num_layers = 3

 


cells = []
for _ in range(num_layers):
  cell = tf.contrib.rnn.LSTMCell(num_units)  # Or GRUCell(num_units)
  cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
  cells.append(cell)
mlstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)


# **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
# ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size] 
# ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
# ** state.shape = [layer_num, 2, batch_size, hidden_size], 
# ** 或者，可以取 h_state = state[-1][1] 作为最后输出
# ** 最后输出维度是 [batch_size, hidden_size]
 
X_reshape = tf.reshape(_X, [-1, timesteps, data_dim])

output, state = tf.nn.dynamic_rnn(mlstm_cell, X_reshape, dtype=tf.float32, initial_state=init_state)
h_state = state[-1][1]

#h_state = output[:, -1, :]  # 或者 h_state = state[-1][1]



##加一个全连接层  输入维数num_units, 输出维数为64
W_fc1 = weight_variable([num_units, 128])
b_fc1 = bias_variable([128])
h_fc1=tf.nn.relu(tf.matmul(h_state,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,1-dropout)


##加一个全连接层  输入维数num_units, 输出维数为64
W_fc2 = weight_variable([128, 64])
b_fc2 = bias_variable([64])
h_fc2=tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2, name='dense')
h_fc2_drop=tf.nn.dropout(h_fc2,1-dropout)


# 开始训练和测试
W = tf.Variable(tf.truncated_normal([64, out_dim], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[out_dim]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_fc2_drop, W) + bias,  name='predict')


# 损失和评估函数
lr = 1e-3



# 损失函数
cross_entropy = -tf.reduce_mean(_y * tf.log(y_pre))


#评估函数
correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(_y,1))
batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
 

# TRAIN StARTS
save_model = 'model_lstm_tf_za'
saver = tf.train.Saver()




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(runEpoch):
        ##每一轮重新打乱数据
        num_train = X_train.shape[0]
        perm = np.arange(num_train)
        np.random.shuffle(perm)  
        #perm = np.random.shuffle(perm)  错误，这样perm就是None了
        contents = X_train[perm]
        labels = y_train[perm]
         
        ##每一轮清零索引
        index_in_epoch = 0
        i=0
        while(i<train_size):
            index_in_epoch = i
            batch = get_next_batch(BS, index_in_epoch, contents, labels)
            i = i + BS
            
            _X_reshape = batch[0].reshape(-1, timesteps, data_dim)
            _y_reshape = batch[1]#.values.reshape(-1,1) 
 
            _, cost, y_prediction = sess.run([train_op, cross_entropy, y_pre], feed_dict={_X:_X_reshape, _y: _y_reshape, dropout: 0.4, batch_size: BS})
              

            step = i/BS
            if step%10==0:
                acc, cost = sess.run([batch_accuracy,cross_entropy], feed_dict={_X:_X_reshape, _y: _y_reshape, dropout: 0.4, batch_size: BS})
                print "Epoch:", e, "step:",step, " cost=", "{:.9f}".format(cost),  "acc=", "{:.9f}".format(acc)

                #print "Epoch:", e, "step:",step, " cost=", "{:.9f}".format(batch_mse),  "Train auc=",  auc(batch[1], y_prediction)
                #print "Epoch:", e, "step:",step, " cost=", "{:.9f}".format(cost),  "Train f1=",  f1_score(batch[1], y_prediction)
         
    
    print("Optimization Finished!")
    save_path = saver.save(sess, save_model)
    print("Model saved in file: %s" % save_path)


_X_train_reshape = X_train.reshape(-1, timesteps, data_dim)
_y_train_reshape = y_train#.values#.reshape(-1,out_dim)


_X_test_reshape = X_test.reshape(-1, timesteps, data_dim)
_y_test_reshape = y_test#.values#.reshape(-1,out_dim)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    #方法1
    #saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    #方法2
    ##注意，不仅仅要给出模型名字model_lstm_tf_za，还一定要给出所在的路径，如果是当前目录下，就用./model_lstm_tf_za，否则会报错Unsuccessful TensorSliceReader constructor
    saver = tf.train.import_meta_graph("./model_lstm_tf_za.meta")
    saver.restore(sess, "./model_lstm_tf_za")
    
    y_predict = sess.run(y_pre, feed_dict={_X: _X_test_reshape, dropout: 0, batch_size: _y_test_reshape.shape[0]})
    #print y_predict.shape
    #print y_test.shape
 
    
#y_predict = np.array( [x[0] for x in y_predict]).reshape(-1,1)
#y_predict = np.where(np.array(y_predict)<0.5,0,1)
#y_test = np.array([x[0] for x in y_test]).reshape(-1,1)

 

y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)




confusion_matrix_1=confusion_matrix(y_test,y_predict)
print  confusion_matrix_1


precision_p = float(confusion_matrix_1[1][1])/float((confusion_matrix_1[0][1] + confusion_matrix_1[1][1]))
recall_p = float(confusion_matrix_1[1][1])/float((confusion_matrix_1[1][0] + confusion_matrix_1[1][1]))
 
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 
F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)
print "LSTM F1:", F1_Score


 
init = tf.global_variables_initializer()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./model_lstm_tf_za.meta")
    saver.restore(sess, "./model_lstm_tf_za")
    graph = tf.get_default_graph()
    
    feature_lstm = graph.get_operation_by_name("dense").outputs[0] 
    test_y_pre, test_feature = sess.run([y_pre, feature_lstm], feed_dict={_X: _X_test_reshape, dropout: 0, batch_size: _y_test_reshape.shape[0]})
    train_y_pre, train_feature = sess.run([y_pre, feature_lstm], feed_dict={_X: _X_train_reshape, dropout: 0, batch_size: _y_train_reshape.shape[0]})
    
print test_y_pre.shape, test_feature.shape

y_train = np.argmax(y_train, axis=1).reshape(-1,1).ravel()

layer_output_train = np.hstack((train_feature,X_train[:, (timesteps-1)*data_dim: ]))
layer_output_test = np.hstack((test_feature,X_test[:, (timesteps-1)*data_dim: ]))


X_train = layer_output_train
X_test = layer_output_test
 

#n_estimators树的数量一般大一点。 max_features 对于分类的话一般特征束的sqrt，auto自动
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features="auto",max_leaf_nodes=None, bootstrap=True)

print  "start fit RF:\n"

clf = clf.fit(X_train, y_train)
  
print clf.score(X_test, y_test)

pred = clf.predict(X_test) 


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
F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)

print F1_Score 