import pandas as pd
  
df = pd.read_csv("idx_new_08_del.csv")

df_cosume=df.groupby([df['pri_acct_no_conv']])

result =  df_cosume.agg({'pri_acct_no_conv':'count'})
result=result[result.pri_acct_no_conv<5]
 
df=df[~df['pri_acct_no_conv'].isin(result.index)]
 
df.to_csv('idx_new_08_more.csv',sep=',') 