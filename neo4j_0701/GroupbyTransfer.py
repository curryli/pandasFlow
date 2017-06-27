import pandas as pd
  
df_cosume = pd.read_csv("transfer_origin.csv", header=None, sep=',')
df_cosume.columns = ['source','target','time','money'] 
 
Grouped = df_cosume.groupby([df_cosume['source'], df_cosume['target']])
#print Grouped['money','hour'].agg(['count','mean','max'])
result =  Grouped.agg({'money':'sum','time':'mean'})

result.to_csv('transfer.csv',sep=',')  
#with open("result.txt",'w') as FILEOUT:
#    for item in result:
#        print>>FILEOUT,item