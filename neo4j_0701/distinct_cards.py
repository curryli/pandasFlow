import pandas as pd
  
df_cosume = pd.read_csv("0701_alltrans.csv", header=None, sep=',')


#print df_cosume.iloc[:, [0]]
distinct_card_list = df_cosume.iloc[:, [0]].stack().unique()  #统计distinct卡号
with open("cardNode.txt",'w') as FILEOUT:
    for card in distinct_card_list:
        print>>FILEOUT,card
#print pd.merge(df_cosume,df_seeds,how='inner')


f = lambda x: str(x)[-4:]
 
df_cosume.columns = ['cardkey','trans_id','money','date','hour','region','trans_md','iscross','src','dst'] 
df_cosume['region'] = df_cosume['region'].map(f)   

distinct_region = df_cosume.iloc[:, [5]].stack().unique()

print distinct_region
with open("regionNode.txt",'w') as FILEOUT:
    for region in distinct_region:
        print>>FILEOUT,region
        
        
df_cosume['hour'] = df_cosume['hour'].map(lambda x: str(x)[:-3])   

distinct_region = df_cosume.iloc[:, [4]].stack().unique()

print distinct_region
with open("hourNode.txt",'w') as FILEOUT:
    for region in distinct_region:
        print>>FILEOUT,region