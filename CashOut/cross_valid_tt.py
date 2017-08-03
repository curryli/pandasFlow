import pandas as pd
import numpy as np 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import time

# from show_confusion_matrix import show_confusion_matrix 
# the above is from http://notmatthancock.github.io/2015/10/28/confusion-matrix.html

df = pd.read_csv("input/idx_new_08.csv")
#df_s=df.drop(['pri_acct_no_conv'], axis =1)#去除'pri_acct_no_conv'列
label='label' # label的值就是二元分类的输出
cardcol= 'pri_acct_no_conv'
df[label].value_counts()

#采样
Fraud = df[df.label == 1]
normal = df[df.label == 0]
#在normal中随机选出与Fraud比例为1:1的样本
fraud_num=10000
#fraud_num=Fraud.shape[0]
ratio_s=14

precision_p=0
recall_p=0

for i in range(0,5):
    print i
    fraud_sample=Fraud.sample(n=fraud_num)

    normal_num=fraud_sample.shape[0]*ratio_s if fraud_sample.shape[0]*ratio_s < normal.shape[0] else normal.shape[0]
    normal_sample=normal.sample(n=normal_num)

    train = pd.concat([fraud_sample,normal_sample], axis = 0)

    x_columns = [x for x in train.columns if x not in [label,cardcol]]
    X = train[x_columns]
    y = train[label]

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
    precision_p = precision_p+float(temp_m[1][1])/float((temp_m[0][1] + temp_m[1][1]))
    recall_p = recall_p+float(temp_m[1][1])/float((temp_m[1][0] + temp_m[1][1]))


print ("Precision:", precision_p/5)
print ("Recall:", recall_p/5)

#10-折交叉验证
import numpy as np
import pandas as pd
import time 
from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold  
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import shuffle  

df = pd.read_csv("input/idx_new_08.csv")
label='label' # label的值就是二元分类的输出
cardcol= 'pri_acct_no_conv'
df[label].value_counts()

#采样
Fraud = df[df.label == 1]
normal = df[df.label == 0]
#在normal中随机选出与Fraud比例为1:1的样本
#fraud_num=10000
fraud_num=Fraud.shape[0]
ratio_s=1 #    1:ratio 表示 Fraud:normal 数据比例

fraud_sample=Fraud.sample(n=fraud_num)
normal_num=fraud_sample.shape[0]*ratio_s if fraud_sample.shape[0]*ratio_s < normal.shape[0] else normal.shape[0]
normal_sample=normal.sample(n=normal_num)

data_train = pd.concat([fraud_sample,normal_sample], axis = 0)

#对数据Fraud 和Normal数据随机打乱顺序
precision_p=0
recall_p=0

Loop_n=1 #循环次数
fold_n=10 #n-折交叉验证：折数

for i in range(0,Loop_n):
    train = shuffle(data_train) 
    x_columns = [x for x in train.columns if x not in [label,cardcol]]
    X = train[x_columns]
    y = train[label]
    
    X=np.array(X)
    y=np.array(y)
    kf= KFold(n_splits=fold_n)  
    kf.get_n_splits(X)#给出K折的折数，输出为2  
    for train_index, test_index in kf.split(X):
        print("TRAIN:",train_index, "TEST:", test_index)  
        X_train,X_test = X[train_index], X[test_index]  
        y_train,y_test = y[train_index], y[test_index]
        
        rf_model = RandomForestClassifier(oob_score=True, random_state=10)
        time1 = time.time()
        rf_model.fit(X_train,y_train)
        time2 = time.time()
        print "rf_model used time: %f sec" % (time2 - time1)  #时间 second
    
        pred_test= rf_model.predict(X_test)  
        temp_m=confusion_matrix(y_test,pred_test)
        precision_p = precision_p+float(temp_m[1][1])/float((temp_m[0][1] + temp_m[1][1]))
        recall_p = recall_p+float(temp_m[1][1])/float((temp_m[1][0] + temp_m[1][1]))

mean_num=Loop_n*fold_n
print ("Precision:", precision_p/mean_num)
print ("Recall:", recall_p/mean_num)