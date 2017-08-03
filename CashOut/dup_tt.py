import numpy as np
import pandas as pd
import time 
from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold  
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import shuffle  
from sklearn.cross_validation import train_test_split

df = pd.read_csv("idx_new_08_del.csv")
label='label' # label的值就是二元分类的输出
cardcol= 'pri_acct_no_conv'
df[label].value_counts()

#采样
Fraud = df[df.label == 1]
normal = df[df.label == 0]
#在normal中随机选出与Fraud比例为1:1的样本
fraud_num=Fraud.shape[0]

#fraud_sample=Fraud.sample(n=fraud_num)
extend_num=fraud_num*200-normal.shape[0]
stack_n=extend_num/normal.shape[0]     #Normal样本重复采样次数
stack_l=extend_num%normal.shape[0]     #Normal样本重复采样余数

normal_train=normal
for i in range(0,stack_n):
    normal_train=pd.concat([normal_train,normal], axis = 0)
normal_train=pd.concat([normal_train,normal.sample(n=stack_l)], axis = 0)
print "extend Normal size: %d" % normal_train.shape[0]

data_train = pd.concat([Fraud,normal_train], axis = 0)

x_columns = [x for x in data_train.columns if x not in [label,cardcol]]
X = data_train[x_columns]
y = data_train[label]

print "datasets size:%d" % X.shape[0]

train_X,test_X, train_y, test_y = train_test_split(X,  
                                               y,  
                                               test_size = 0.2,  
                                               random_state = 0) 

print "train size:%d" % train_X.shape[0] 
print "test size:%d" % test_X.shape[0] 

rf_model = RandomForestClassifier(oob_score=True, random_state=10)
time1 = time.time()
rf_model.fit(train_X,train_y)
time2 = time.time()
print "rf_model used time: %f sec" % (time2 - time1)  #时间 second

pred_test= rf_model.predict(test_X)  
temp_m=confusion_matrix(test_y,pred_test)
print temp_m
precision_p = float(temp_m[1][1])/float((temp_m[0][1] + temp_m[1][1]))
recall_p = float(temp_m[1][1])/float((temp_m[1][0] + temp_m[1][1]))

print ("Precision:", precision_p)
print ("Recall:", recall_p)