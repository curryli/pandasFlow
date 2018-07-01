# -*-coding:utf-8-*-
#https://blog.csdn.net/jackxu8/article/details/71159315

# conda uninstall mkl=2018
# conda install mkl=2017

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
import theano

data = pd.read_csv('radon.csv')
data['log_radon'] = data['log_radon'].astype(theano.config.floatX)
county_names = data.county.unique()
county_idx = data.county_code.values

n_counties = len(data.county.unique())
print(n_counties)

# 模型需要使用的数据的一小部分
data[['county', 'log_radon', 'floor']].head()


with pm.Model() as unpooled_model:

    # 每个国家的独立参数
    alpha = pm.Normal('alpha', 0, sd=100, shape=n_counties)
    beta = pm.Normal('beta', 0, sd=100, shape=n_counties)

    # 模型误差
    eps = pm.HalfCauchy('eps', 5)

    # radon含量的数学模型
    radon_est = alpha[county_idx] + beta[county_idx]*data.floor.values

    # Data likelihood
    # 均值就是待预测的radon含量
    # 方差就是测量误差eps
    # 并给定观测值（测量值）
    y = pm.Normal('y', mu=radon_est, sd=eps, observed=data.log_radon)

with unpooled_model:
    unpooled_trace = pm.sample(5000)

pm.traceplot(unpooled_trace)


with pm.Model() as hierarchical_model:

    # 超参数
    # HalfCauchy 柯西半连续型
    # Normal 正态型
    mu_alpha = pm.Normal('mu_alpha', mu=0., sd=100**2)
    sigma_alpha = pm.HalfCauchy('sigma_alpha', 5)
    mu_beta = pm.Normal('mu_beta', mu=0., sd=100**2)
    sigma_beta = pm.HalfCauchy('sigma_beta', 5)

    # 每个国家的参数均服从同一个正态分布
    alpha = pm.Normal('alpha', mu=mu_alpha, sd=sigma_alpha, shape=n_counties)
    beta = pm.Normal('beta', mu=mu_beta, sd=sigma_beta, shape=n_counties)

    # 模型误差
    eps = pm.HalfCauchy('eps', 5)

    # radon含量的模型
    radon_est = alpha[county_idx] + beta[county_idx] * data.floor.values

    # Data likelihood
    # 均值就是待预测的radon含量
    # 方差就是测量误差eps
    # 并给定观测值（测量值）
    radon_like = pm.Normal('radon_like', mu=radon_est, sd=eps, observed=data.log_radon)

with hierarchical_model:
    hierarchical_trace = pm.sample(5000)