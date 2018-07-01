# -*-coding:utf-8-*-
#https://blog.csdn.net/gnahznib/article/details/70244175  概率图模型之：贝叶斯网络
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

# 通过边来定义贝叶斯模型
model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

# 定义条件概率分布
# variable：变量
# variable_card：基数
# values：变量值
# evidence：受哪些变量影响
cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.6, 0.4]])
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.7, 0.3]])
cpd_g = TabularCPD(variable='G', variable_card=3,
                   values=[[0.3, 0.05, 0.9,  0.5],
                           [0.4, 0.25, 0.08, 0.3],
                           [0.3, 0.7,  0.02, 0.2]],
                  evidence=['I', 'D'],
                  evidence_card=[2, 2])

cpd_s = TabularCPD(variable='S', variable_card=2,
                   values=[[0.95, 0.2],
                           [0.05, 0.8]],
                   evidence=['I'],
                   evidence_card=[2])


cpd_l = TabularCPD(variable='L', variable_card=2,
                   values=[[0.1, 0.4, 0.99],
                           [0.9, 0.6, 0.01]],
                   evidence=['G'],
                   evidence_card=[3])

# 将有向无环图与条件概率分布表关联
model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)

# 验证模型：检查网络结构和CPD，并验证CPD是否正确定义和总和为1
model.check_model()

#获取上述代码构建的概率图模型：
print(model.get_cpds())

#获取结点G的概率表：
print(model.get_cpds('G'))
#获取整个贝叶斯网络的局部依赖
print(model.local_independencies(['D', 'I', 'S', 'G', 'L']))

from pgmpy.inference import VariableElimination
infer = VariableElimination(model)
print(infer.query(['G']) ['G'])

#计算P(G|D=0,I=1)的条件分布概率
print(infer.query(['G'], evidence={'D': 0, 'I': 1}) ['G'])

print(infer.map_query('G'))