# -*- coding: utf-8 -*-


#==============================================================================
# 相同交易卡号数据按交易时间往前迭代合并，得到时间区间的特征数据
#输入：
#    交易卡分散的数据，格式：卡号 特征 标签
#输出：交易时间 
#    特征1 特征2...特征n 最近一笔交易标签...
#==============================================================================
import pandas as pd
from pandas import DataFrame
import numpy as np
df = pd.read_csv("idx_new_08_sort_test.csv")
df=df.sort_values(by=['pri_acct_no_conv','tfr_dt_tm']) #按pri_acct_no_conv列的升序排序 
#df.index=range(0,df.shape[0]) 

#df=df.head()
data=np.array(df.iloc[:,1:], dtype=np.int64)
#获取所有的卡号 及索引号


df['pri_acct_no_conv'].index=range(0,card_c.shape[0]) #排序后的卡号修改索引
card_c_d = df['pri_acct_no_conv'].drop_duplicates()#删除重复交易卡号

print card_c_d

#遍历合并 每张卡号的第一个索引
c_index=card_c_d.index
c_index_list = list(c_index)


for cn in c_index_list:
    temp=[]
    count=np.sum(df['pri_acct_no_conv']==card_c_d[cn]) #计算某张卡的行数  card_c_d[cn]是某个卡号
    for j in range(cn,cn+count):
        temp=np.hstack((temp,data[j][0:-1])) #同一卡号的特征合并,标签不合并
        temp_s=temp
        temp_s=np.hstack((temp_s,data[j][-1]))  #保留最近的标签
        temp_s=temp_s.reshape(1,len(temp_s))
        data_temp = DataFrame(temp_s)
        data_temp.to_csv('data_5_rows.csv',index=False,header=False,mode='a')
        
        
        
#a = np.array((1,2,3))  
#b = np.array((2,3,4))  
#c = np.hstack((a,b))
#print c
# 
#print DataFrame(c) 

