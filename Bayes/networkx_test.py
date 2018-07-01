# -*-coding:utf-8-*-
import networkx as nx
from collections import Counter
import urllib
import matplotlib.pyplot as plt
g=nx.Graph()
#g=nx.DiGraph()
g.add_edge('Card_a','Card_b')
g.add_edge('Card_a','Card_c')
g.add_edge('Card_a','POS_d')
g.add_edge('Card_a','POS_e')
g.add_edge('Card_c','POS_d')


g.node['Card_a']['type']="card"
g.node['Card_b']['type']="card"
g.node['Card_c']['type']="card"
g.node['POS_d']['type']="POS"
g.node['POS_e']['type']="POS"


# g.add_edge('a','b', 'type' = "Transfer")

g['Card_a']['Card_b']['type']="Transfer"
g['Card_a']['Card_c']['type']="Transfer"
g['Card_a']['POS_e']['type']="Spend"
g['Card_a']['POS_d']['type']="Withdraw"
g['Card_c']['POS_d']['type']="Withdraw"


flt_func = lambda d: d['type'] == "POS"

#属性为 "POS" 的节点列表
pos_nodes = [n for n, d in g.nodes(data=True) if flt_func(d)]

#print g.node["a"]['type']


#连接两个card节点以上的POS 节点
use_pos = []
for n in pos_nodes:
    type_list = [g.node[i]['type'] for i in g.neighbors(n)]
    if Counter(type_list)["card"]>=2:
        use_pos.append(n)

print use_pos

color_func = lambda n: "b" if n in use_pos else "r"
colors = [color_func(n) for n in g.nodes()]

nx.draw(g, with_labels=True, node_color=colors)

plt.show()


# nx.draw_spring(g)
# #nx.draw_networkx_edge_labels(g,pos,font_size=10,alpha=0.5,rotate=True)
# nx.draw_networkx_labels(g,pos,font_size=10,alpha=0.5,rotate=False)
# #net.draw(part_G)
# #plt.savefig("youxiangtu.png")
# plt.show()