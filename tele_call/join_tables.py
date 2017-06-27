# -*- coding: utf-8 -*-
import pandas as pd
  
df_call = pd.read_csv("call_3.csv", header=0, names=["phone","group_0","name","call_1","call_2"], dtype={"phone": str, "group_0": str, "name": str} , sep=',')
#df_call = df_call.drop_duplicates(subset=["phone","name"], keep='first', inplace=True)
df_call = df_call.drop_duplicates(subset=["phone"], keep='first', inplace=False)
print df_call.shape[0]
#print df_call
#df_call[["phone"]].astype(str)
#print df_call.loc[0:5]
#print df_call.iloc[0:5,0:2]



df_element3 = pd.read_csv("element3.csv", header=0, names=["phone","name","idcard","phonetype","validate","status","onlinetime_2"], dtype={"phone": str, "idcard": str, "name": str} , sep=',')
df_element3 = df_element3.drop_duplicates(subset=["phone"], keep='first', inplace=False)
#print df_element3.count()
print df_element3.shape[0]
#print df_element3.loc[11:13][["name"]]


df_external = pd.read_csv("external.csv", header=0, names=["phone","isbad","state","onlinetime_1","location"], dtype={"phone": str,"location": str} , sep=',')
df_external = df_external.drop_duplicates(subset=["phone"], keep='first', inplace=False)
print df_external.shape[0]
#print df_external

df_bank = pd.read_csv("phone_bank.csv", header=0, names=["phone","bankcard"], dtype={"phone": str,"bankcard": str} , sep=',')
df_bank = df_bank.drop_duplicates(subset=["phone"], keep='first', inplace=False)
#print df_bank.shape[0]
#print df_external


df_call_element3 = pd.merge(left=df_call, right=df_element3, how='left', left_on='phone', right_on='phone')
#print("aaaaaaaaaaaaaaaaaaaaaaa")
#print df_call_element3.loc[0:5] 


df_call_element3_external = pd.merge(left=df_call_element3, right=df_external, how='left', left_on='phone', right_on='phone')
#print("bbbbbbbbbbbbbbbbbbbbb")
 
df_all = pd.merge(left=df_call_element3_external, right=df_bank, how='left', left_on='phone', right_on='phone')

df_all = df_all.fillna("NAN") 

#df_all["call_result"]

df_all = df_all[["phone","group_0","name_x","idcard","bankcard","location","phonetype","validate","status","onlinetime_1","onlinetime_2","state","isbad","call_1","call_2"]]


def judge(x, y):
    if(x=='0' or y=='0'):
        return 0
    else:
        return 1

df_all["call_result"] = map(lambda x,y : judge(x, y), df_all['call_1'], df_all['call_2'])

#df_all.to_csv("result.csv")
#print df_all.loc[0:5] 



df_card_md5 = pd.read_csv("police_card.csv", header=0, names=["ori_card","md5_card"], dtype={"ori_card": str,"md5_card": str} , sep=',')
#print df_card_md5.columns

df_card_trans = pd.read_csv("card_trans.csv", header=0, names=["md5_card","pdate","trans_id","trans_at"] , sep=',')
#print df_card_trans.shape[0]

#md5_minpdate_df = df_card_trans[["md5_card","pdate"]].groupby(df_card_trans["md5_card"]).agg(['min'])
group_minpdate_df = df_card_trans.groupby(df_card_trans["md5_card"])["pdate"].agg(['min','count'])

md5_minpdate_df = group_minpdate_df.reset_index() 

card_minpdate_df = pd.merge(left=df_card_md5, right=md5_minpdate_df, how='left', left_on="md5_card", right_on="md5_card")[["ori_card","min",'count']]
#print card_minpdate_df.loc[0:5]
 
result_df = pd.merge(left=df_all, right=card_minpdate_df, how='left', left_on="bankcard", right_on="ori_card") 
result_df["idcard"] = result_df["idcard"].map(lambda x: r"'"+x) 
result_df["bankcard"] = result_df["bankcard"].map(lambda x: r"'"+x) 
result_df = result_df.drop("ori_card",axis=1, inplace=False)

#result_df.to_csv("result.csv")
for i in range(0,14):
    print "call_1: " + str(i) 
    print result_df[result_df["call_1"]==str(i)].groupby(result_df["call_2"]).agg('count')["call_2"]
    print "\n\n\n"
