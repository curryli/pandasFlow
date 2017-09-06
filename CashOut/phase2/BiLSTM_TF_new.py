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

import os                          #python miscellaneous OS system tool
#os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
 
labelName="label" 
cardName = "pri_acct_no_conv" 
runEpoch=50
 
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

 
# Parameters
learning_rate = 0.01
 

# Network Parameters
n_input = data_dim # MNIST data input (img shape: 28*28)
n_steps = timesteps # timesteps
n_hidden = 256 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def BiRNN(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
#    try:
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                           dtype=tf.float32)
#    except Exception: # Old TensorFlow version only returns outputs not states
#        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
#                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    pred =  tf.matmul(outputs[-1], weights['out']) + biases['out']
    y_pred = tf.argmax(pred,1)
    return pred, y_pred
    




pred, y_pred = BiRNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


 
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
            #print _X_reshape.shape
            #print _y_reshape.shape
             
            sess.run(optimizer, feed_dict={x:_X_reshape, y: _y_reshape})
              

            step = i/BS
            if step%10==0:
                batch_accuracy, batch_mse = sess.run([accuracy,cost], feed_dict={x:_X_reshape, y: _y_reshape})
                print "Epoch:", e, "step:",step, " cost=", "{:.9f}".format(batch_mse),  "batch_accuracy=", "{:.9f}".format(batch_accuracy)

                #print "Epoch:", e, "step:",step, " cost=", "{:.9f}".format(batch_mse),  "Train auc=",  auc(batch[1], y_prediction)
                #print "Epoch:", e, "step:",step, " cost=", "{:.9f}".format(cost),  "Train f1=",  f1_score(batch[1], y_prediction)
        
    # 计算测试数据的准确率
#    _X_test_reshape = X_test.reshape(-1, timesteps, data_dim)
#    _y_test_reshape = y_test#.values#.reshape(-1,out_dim)
#    print "test accuracy %g"% sess.run(accuracy, feed_dict={_X: _X_test_reshape, _y: _y_test_reshape, dropout: 0, batch_size: _y_test_reshape.shape[0]})
    
    print("Optimization Finished!")
    save_path = saver.save(sess, save_model)
    print("Model saved in file: %s" % save_path)


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
    
    accuracy_train = sess.run(accuracy, feed_dict={x:_X_reshape, y: _y_reshape})
    
    pred_test, accuracy_test = sess.run([pred,accuracy], feed_dict={x:_X_test_reshape, y: _y_test_reshape})
     
    print "accuracy_train: ", accuracy_train
    print "accuracy_test", accuracy_test
 
  
print "pred_test.shape: ", pred_test.shape
 
print "y_test.shape: ", y_test.shape

y_predict = np.argmax(pred_test, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

print "y_pred.shape: ", y_predict.shape
print "y_test.shape: ", y_test.shape


confusion_matrix=confusion_matrix(y_test,y_predict)
print  confusion_matrix


precision_p = float(confusion_matrix[1][1])/float((confusion_matrix[0][1] + confusion_matrix[1][1]))
recall_p = float(confusion_matrix[1][1])/float((confusion_matrix[1][0] + confusion_matrix[1][1]))
 
print ("Precision:", precision_p) 
print ("Recall:", recall_p) 
F1_Score = 2*precision_p*recall_p/(precision_p+recall_p)
print F1_Score


#[[183137    102]
# [   626    784]]
#('Precision:', 0.8848758465011287)
#('Recall:', 0.5560283687943263)
#0.682926829268