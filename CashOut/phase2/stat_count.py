# -*- coding: utf-8 -*-
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import shuffle  
from sklearn.externals import joblib 
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

start_time = datetime.datetime.now()

  
label='label' # label的值就是二元分类的输出
cardcol= 'pri_acct_no_conv'
time = "tfr_dt_tm"



################################################# 
 
 
reader = pd.read_csv("FE_train_data.csv", low_memory=False, iterator=True)
     
loop = True
chunkSize = 100000 
chunks = []
i = 0
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
        if (i%5)==0:
            print i
        i = i+1
   
    except StopIteration:
        loop = False
        print "Iteration is stopped."
df_all = pd.concat(chunks, ignore_index=True)


Grouped = df_all.groupby([df_all[cardcol]])
cnt_pair = Grouped.agg({'label':'count'})
print type(cnt_pair )
print cnt_pair[(cnt_pair.label>0)&(cnt_pair.label<=5)].shape[0]
print cnt_pair[(cnt_pair.label>5)&(cnt_pair.label<=10)].shape[0]
print cnt_pair[(cnt_pair.label>10)&(cnt_pair.label<=15)].shape[0]
print cnt_pair[(cnt_pair.label>15)&(cnt_pair.label<=20)].shape[0]
print cnt_pair[(cnt_pair.label>20)&(cnt_pair.label<=25)].shape[0]
print cnt_pair[(cnt_pair.label>30)&(cnt_pair.label<=35)].shape[0]
print cnt_pair[(cnt_pair.label>35)&(cnt_pair.label<=40)].shape[0]
print cnt_pair[(cnt_pair.label>45)&(cnt_pair.label<=50)].shape[0]
print cnt_pair[(cnt_pair.label>50)&(cnt_pair.label<=55)].shape[0]
print cnt_pair[(cnt_pair.label>55)&(cnt_pair.label<=60)].shape[0]
print cnt_pair[(cnt_pair.label>60)&(cnt_pair.label<=65)].shape[0]
print cnt_pair[(cnt_pair.label>65)&(cnt_pair.label<=70)].shape[0]
print cnt_pair[(cnt_pair.label>70)&(cnt_pair.label<=75)].shape[0]
print cnt_pair[(cnt_pair.label>75)&(cnt_pair.label<=80)].shape[0]
print cnt_pair[(cnt_pair.label>80)&(cnt_pair.label<=85)].shape[0]
print cnt_pair[(cnt_pair.label>85)&(cnt_pair.label<=90)].shape[0]
print cnt_pair[(cnt_pair.label>90)&(cnt_pair.label<=95)].shape[0]
print cnt_pair[(cnt_pair.label>95)&(cnt_pair.label<=100)].shape[0]
print cnt_pair[(cnt_pair.label>100)].shape[0]
