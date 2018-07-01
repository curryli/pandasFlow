from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#https://blog.csdn.net/lpsl1882/article/details/74457568
import numpy as np
import pandas as pd
import edward as ed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from edward.models import Bernoulli, Normal, Empirical
import matplotlib.pyplot as plt

data = pd.read_csv('insteval.csv')
data['dcodes'] = data['d'].astype('category').cat.codes
data['deptcodes'] = data['dept'].astype('category').cat.codes
data['s'] = data['s'] - 1

train = data.sample(frac=0.8)
test = data.drop(train.index)

s_train = train['s'].values
d_train = train['dcodes'].values
dept_train = train['deptcodes'].values
y_train = train['y'].values
service_train = train['service'].values
n_obs_train = train.shape[0]

s_test = test['s'].values
d_test = test['dcodes'].values
dept_test = test['deptcodes'].values
y_test = test['y'].values
service_test = test['service'].values
n_obs_test = test.shape[0]

n_s = max(s_train) + 1  # number of students
n_d = max(d_train) + 1  # number of instructors
n_dept = max(dept_train) + 1  # number of departments
n_obs = train.shape[0]  # number of observations

print("Number of students: {}".format(n_s))
print("Number of instructors: {}".format(n_d))
print("Number of departments: {}".format(n_dept))
print("Number of observations: {}".format(n_obs))

# 数据输入
s_ph = tf.placeholder(tf.int32, [None])#学生编号category
d_ph = tf.placeholder(tf.int32, [None])#教师编号category
dept_ph = tf.placeholder(tf.int32, [None])#部门编号category
service_ph = tf.placeholder(tf.float32, [None])#二值项，固定特征

#固定特征参数项
mu = tf.Variable(tf.random_normal([]))#Bf
service = tf.Variable(tf.random_normal([]))#beta

#随机特征截距的参数
sigma_s = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))#学生Bs的方差
sigma_d = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))#教师Bd的方差
sigma_dept = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))#部门Bdept方差

# 随机特征截距
eta_s = Normal(loc=tf.zeros(n_s), scale=sigma_s * tf.ones(n_s))
eta_d = Normal(loc=tf.zeros(n_d), scale=sigma_d * tf.ones(n_d))
eta_dept = Normal(loc=tf.zeros(n_dept), scale=sigma_dept * tf.ones(n_dept))


#随机特征项+固定特征项
yhat = tf.gather(eta_s, s_ph) + \
    tf.gather(eta_d, d_ph) + \
    tf.gather(eta_dept, dept_ph) + \
    mu + service * service_ph     #这里tf.gather实际作用是样本采样  https://blog.csdn.net/guotong1988/article/details/53172882
y = Normal(loc=yhat, scale=tf.ones(n_obs))



q_eta_s = Normal(
    loc=tf.get_variable("q_eta_s/loc", [n_s]),
    scale=tf.nn.softplus(tf.get_variable("q_eta_s/scale", [n_s])))
q_eta_d = Normal(
    loc=tf.get_variable("q_eta_d/loc", [n_d]),
    scale=tf.nn.softplus(tf.get_variable("q_eta_d/scale", [n_d])))
q_eta_dept = Normal(
    loc=tf.get_variable("q_eta_dept/loc", [n_dept]),
    scale=tf.nn.softplus(tf.get_variable("q_eta_dept/scale", [n_dept])))

latent_vars = {
    eta_s: q_eta_s,
    eta_d: q_eta_d,
    eta_dept: q_eta_dept}
data = {
    y: y_train,
    s_ph: s_train,
    d_ph: d_train,
    dept_ph: dept_train,
    service_ph: service_train}
inference = ed.KLqp(latent_vars, data)

yhat_test = ed.copy(yhat, {
    eta_s: q_eta_s.mean(),
    eta_d: q_eta_d.mean(),
    eta_dept: q_eta_dept.mean()})


inference.initialize(n_print=20, n_iter=100)
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  # Update and print progress of algorithm.
  info_dict = inference.update()
  inference.print_progress(info_dict)

  t = info_dict['t']
  if t == 1 or t % inference.n_print == 0:
    # Make predictions on test data.
    yhat_vals = yhat_test.eval(feed_dict={
        s_ph: s_test,
        d_ph: d_test,
        dept_ph: dept_test,
        service_ph: service_test})

    # Form residual plot.
    # plt.title("Residuals for Predicted Ratings on Test Set")
    # plt.xlim(-4, 4)
    # plt.ylim(0, 800)
    # plt.hist(yhat_vals - y_test, 75)
    # plt.show()