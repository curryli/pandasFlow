# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np

df = pd.read_csv("label_all_07.csv")
df.columns
fraud=df[df['label']==1]
normal=df[df['label']==0]

a=pd.value_counts(fraud['pri_acct_no_conv'].values, sort=False)

f_card=fraud['pri_acct_no_conv'].drop_duplicates()
overlap_normal=normal[normal['pri_acct_no_conv'].isin(f_card)]

b=pd.value_counts(overlap_normal['pri_acct_no_conv'].values, sort=False)

c=pd.concat([b,a],axis=1)
c=c.fillna(0)
c['ratio']=c[1].values/(c[0].values+c[1].values)

c.to_csv('card_0_1.csv',index=True, header=True,sep=',',index_label='pri_acct_no_conv')