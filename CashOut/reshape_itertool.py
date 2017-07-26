# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:41:16 2017

@author: yuwei
"""

import pandas as pd                #data analysing tool for python
import matplotlib.pyplot as plt    #data visualising tool for python
import numpy as np
from IPython.display import display
from sklearn.preprocessing import StandardScaler                   # for normalization of our data        #package allowing keras to work with python

from sklearn.utils import shuffle                                  # shuffling our own made dataset                           # linear layer stacks model for keras
from itertools import groupby
from operator import itemgetter
 
import time

labelName="label"
cardName="pri_acct_no_conv"
timeName = "tfr_dt_tm"


global transcationsPerRow
transcationsPerRow=10
start = time.clock()

allsamples = pd.read_csv('sorted_test.csv')
 
## sort allsamples by card/time if needed
print "Sorting according to card and time ..."
allsamples = allsamples.sort_values(by=[cardName, timeName])#sort_index
## end sort

global dropcardlabel
dropcardlabel=allsamples.drop(labelName, axis = 1, inplace=False)
 
dropcardlabel = np.array(dropcardlabel)
#print dropcardlabel

with open(r'result.csv', 'w') as Fout:
    for _, item in groupby(dropcardlabel, key=itemgetter(0)):
        #print len(list(item))
        #print list(item)
        nparr_list = list(item)
        line_result = [list(i)[1:] for i in nparr_list]
        print>>Fout,  line_result
    
 
