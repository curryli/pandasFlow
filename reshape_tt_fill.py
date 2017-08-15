# -*- coding: utf-8 -*-
#==============================================================================
# 相同交易卡号数据按交易时间往前迭代合并，得到时间区间的特征数据，并保留最近10笔交易
#输入：
#    交易卡分散的数据，格式：卡号 特征 标签
#输出：交易时间 
#    特征1 特征2...特征n 最近一笔交易标签...
#==============================================================================
import pandas as pd
from pandas import DataFrame
import numpy as np
df = pd.read_csv("idx_new_08_del.csv")
df=df.sort_values(by=['pri_acct_no_conv','tfr_dt_tm']) #°´pri_acct_no_convÁÐµÄÉýÐòÅÅÐò 
#df=df.head()
#df.index=range(0,df.shape[0]) 

#df=df.head()
data=np.array(df.iloc[:,1:], dtype=np.int64)
#获取所有的卡号 及索引号

card_c=df['pri_acct_no_conv']
df['pri_acct_no_conv'].index=range(0,card_c.shape[0]) #排序后的卡号修改索引
card_c_d = df['pri_acct_no_conv'].drop_duplicates()#删除重复交易卡号

print card_c_d

#±éÀúºÏ²¢ Ã¿ÕÅ¿¨ºÅµÄµÚÒ»¸öË÷Òý
c_index=card_c_d.index
c_index_list = list(c_index)

#合并后保留特征长度
sequnece_l=(df.shape[1]-2)*5

df.shape[1]
for cn in c_index_list:
    temp=np.zeros((sequnece_l,), dtype=np.float) 
    count=np.sum(df['pri_acct_no_conv']==card_c_d[cn]) #¼ÆËãÄ³ÕÅ¿¨µÄÐÐÊý  card_c_d[cn]ÊÇÄ³¸ö¿¨ºÅ
    for j in range(cn,cn+count):
        temp=np.hstack((temp,data[j][0:-1])) #同一卡号的特征合并,标签不合并
        temp_s=temp[-sequnece_l:] #截取最近10笔交易
        temp_s=np.hstack((temp_s,data[j][-1])) #添加最近标签
        temp_s=temp_s.reshape(1,len(temp_s))
        data_temp = DataFrame(temp_s)
        data_temp.to_csv('test1_5.csv',index=False,header=False,mode='a')
        
        

