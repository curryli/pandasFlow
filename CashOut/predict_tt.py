 
 
import pandas as pd
import numpy as np 
#import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
# from show_confusion_matrix import show_confusion_matrix 
# the above is from http://notmatthancock.github.io/2015/10/28/confusion-matrix.html

df = pd.read_csv("idx_new_08.csv")

#df.isnull().sum()

df_s=df.drop(['pri_acct_no_conv','tfr_dt_tm','trans_at','total_disc_at'], axis =1)

#o_fea=df_s.ix[:,0:-1]
o_fea=df_s.iloc[:,0:-2]

attr_fea=o_fea.describe()

#print type(attr_fea)
  
v_features = attr_fea.iloc[:,0:-2].columns
#print v_features

attr_fea_T = attr_fea.T["std"].index        
#print attr_fea_T

for col in v_features:
    itemp=attr_fea[col]["std"]
    if(itemp==0.0):
        print(col)
        




#df_s=df_s.drop(['dis_47','dis_52','dis_53','dis_57','dis_59','dis_61','dis_64','dis_65'], axis =1)

 
#Select only the anonymized features.
top = df_s.shape[1]-1
v_features = df_s.iloc[:,0:-2].columns
 
plt.figure(figsize=(12,top*4))
gs = gridspec.GridSpec(top, 1)
for i, cn in enumerate(df_s[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(df_s[cn][df_s.label == 1], hist=True,kde=False,bins=20)
    sns.distplot(df_s[cn][df_s.label == 0], hist=True,kde=False,bins=20)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.savefig("p-kde-20.png")
#plt.show()
 