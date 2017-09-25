# -*- coding: utf-8 -*-
#==============================================================================
# 相同交易卡号数据按交易时间往前迭代合并，得到时间区间的特征数据，并保留最近m笔交易
#输入：
#    交易卡分散的数据，格式：卡号 特征 标签
#输出：交易时间 
#    交易卡号 （特征1 特征2...特征n）*m 最近一笔交易标签...
#==============================================================================

import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import time

df = pd.read_csv("weika_GBDT_07.csv")
df=df.sort_values(by=['pri_acct_no_conv','tfr_dt_tm']) #°´pri_acct_no_convÁÐµÄÉýÐòÅÅÐò 
#df=df.head(2000)
df_cosume=df.groupby([df['pri_acct_no_conv']])

cols_count =  df_cosume.agg({'pri_acct_no_conv':'count'})

data=np.array(df.iloc[:,1:], dtype=np.int32)
#获取所有的卡号 及索引号

card_c=df['pri_acct_no_conv']
df['pri_acct_no_conv'].index=range(0,card_c.shape[0]) #排序后的卡号修改索引
card_c_d = df['pri_acct_no_conv'].drop_duplicates()#删除重复交易卡号

c_index=card_c_d.index
c_index_list = list(c_index)

#合并后保留特征长度,最近m笔交易
m=5
sequnece_l=(df.shape[1]-2)*m #交易记录特征长度
trans_l=sequnece_l+2  #交易记录长度
#生成新的列标签
colstemp=[]
for i in range(0,m):
    colstemp=np.hstack((colstemp,df.columns.values[1:-1]+'-'+str(i+1)))
    
colstemp=np.hstack((df.columns.values[0],colstemp))    
colstemp=np.hstack((colstemp,df.columns.values[-1]))

sf_name='convert5_weika_GBDT_07.csv'#保存结果数据文件名
if(os.path.exists(sf_name)):
    os.remove(sf_name)

start=time.time()
colstemp=colstemp.reshape(1,len(colstemp))  #添加列名作为numpy数组的第一行 
colsName=DataFrame(colstemp)
colsName.to_csv(sf_name,index=False,header=False,mode='a') 
  
fea_array=np.zeros((trans_l,), dtype=np.int32)
for cn in c_index_list:
    temp=np.zeros((sequnece_l,), dtype=np.int32)
    count=cols_count.pri_acct_no_conv[cols_count.index==card_c_d[cn]]
    for j in range(cn,cn+count):
        if j%100 ==0:
            print j
        temp=np.hstack((temp,data[j][0:-1])) #同一卡号的特征合并,标签不合并
        temp_s=temp[-sequnece_l:] #截取最近10笔交易
        temp_s=np.hstack((temp_s,data[j][-1])) #添加最近标签
        temp_s=np.hstack((card_c_d[cn],temp_s))#添加卡号
        temp_s=temp_s.reshape(1,len(temp_s)) 
        
        fea_array=np.vstack((fea_array,temp_s)) #累加数据
        if j%50 ==0 or j+1==data.shape[0]:
            data_temp = DataFrame(fea_array[1:])
            data_temp.to_csv(sf_name,index=False,header=False,mode='a')
            fea_array=np.zeros((trans_l,), dtype=np.int32)        
end=time.time()
print "used time:%f" %(end-start)