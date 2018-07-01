
# coding: utf-8
#http://python.jobbole.com/85991/  用 Python 进行贝叶斯模型建模

from IPython.display import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats
import scipy.optimize as opt
import statsmodels.api as sm

plt.style.use('bmh')
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', 
          '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']

if __name__=='__main__':
    messages = pd.read_csv('../data/hangout_chat_data.csv')

    with pm.Model() as model:
        mu = pm.Uniform('mu', lower=0, upper=60)
        likelihood = pm.Poisson('likelihood', mu=mu, observed=messages['time_delay_seconds'].values)

        start = pm.find_MAP()  #MAP 代表最大后验估计，它帮助 MCMC 采样器寻找合适的采样起始点  是一个固定的u值
        step = pm.Metropolis()  #一种MCMC
        trace = pm.sample(10, step, start=start, progressbar=True)

