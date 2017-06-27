import pandas as pd
  
df_cosume = pd.read_csv("0701_alltrans.csv", header=None, sep=',')
df_cosume.columns = ['cardkey','trans_id','money','date','hour','region','trans_md','iscross','src','dst'] 

df_cosume['hour'].groupby([df_cosume['cardkey'], df_cosume['trans_id']])
 
df_cosume['region'] = df_cosume['region'].map(lambda x: str(x)[-4:])   
distinct_region = df_cosume.iloc[:, [5]].stack().unique()
 
        
df_cosume['hour'] = df_cosume['hour'].map(lambda x: float(str(x)[:-3]))   
distinct_hour = df_cosume.iloc[:, [4]].stack().unique()
 
#print df_cosume.groupby([df_cosume['cardkey'], df_cosume['trans_id'], df_cosume['region']])['hour'].mean()

Grouped = df_cosume.groupby([df_cosume['cardkey'], df_cosume['trans_id'], df_cosume['region']])
#print Grouped['money','hour'].agg(['count','mean','max'])
result =  Grouped.agg({'money':'sum','hour':'mean'})

result.to_csv('result.csv',sep=',')  
#with open("result.txt",'w') as FILEOUT:
#    for item in result:
#        print>>FILEOUT,item