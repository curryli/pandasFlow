
# coding: utf-8

# In[1]:


#get_ipython().magic(u'matplotlib inline')
# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
from sklearn.utils import shuffle 
# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
import numpy as np
np.random.seed(sum(map(ord, "categorical")))

# Next, we'll load the Alldata dataset, which is in current directory
Alldata = pd.read_csv("idx_new_08.csv") # the Alldata  dataset is now a Pandas DataFrame
Alldata = shuffle(Alldata)
#Alldata = Alldata.sample(frac=0.0005, replace=True)




#Alldata = Alldata[Alldata['label']==1]


print Alldata.shape

#Alldata = pd.read_csv("idx_new_08.csv") # the Alldata  dataset is now a Pandas DataFrame
# Let's see what's in the Alldata data - Jupyter notebooks print the result of the last thing you do

# Press shift+enter to execute this cell


# In[2]:


#Alldata=Alldata[['trans_md','card_media','card_attr','pos_entry_md_cd','term_tp','label']]
Alldata=Alldata[['card_media','card_attr','pos_entry_md_cd','term_tp','label']]
Alldata.info()
#Alldata = Alldata.replace(0.0, 99)
#Alldata['label'] = Alldata['label'].replace(0.0, 2)
print Alldata

#Alldata.plot(kind='scatter', x="card_media", y="pos_entry_md_cd" ).get_figure().show()

#sns.pairplot(Alldata, hue='label', size=3, diag_kind="kde") 

#sns.jointplot(x="card_media", y="pos_entry_md_cd",  data=Alldata) 


# In[3]:

#sns.swarmplot(x="card_media", y="pos_entry_md_cd", hue="label", data=Alldata).get_figure().savefig("sns.png") 


#ax = sns.kdeplot(Alldata['card_media'][AlldaAlldata['label']ta["label"]==1],color='r', shade=True)
#ax = sns.kdeplot(Alldata['card_media'][Alldata["label"]==0],color='b', shade=True)

 
#x, y = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=300).T
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
#sns.kdeplot(Alldata['card_media'], Alldata['pos_entry_md_cd'], cmap="Blues_d", shade=True);
#sns.kdeplot(Alldata['card_media'], Alldata['pos_entry_md_cd'], cmap=cmap, shade=False, n_levels=10).get_figure().savefig("card_media_pos_entry_md_cd_0.png")

g = sns.PairGrid(Alldata) 
try:
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, cmap=cmap, n_levels=20)
except ValueError,e:
    print e.message

g.savefig("snskdeplot.png")


#如果出现   ValueError: zero-size array to reduction operation minimum which has no identity  说明某一列可能只有1种取值
#sns.kdeplot(Alldata['term_tp'], Alldata['pos_entry_md_cd'], cmap=cmap, shade=False, n_levels=10).get_figure().savefig("card_media_pos_entry_md_cd_0.png")

 
