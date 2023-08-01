import eel

import multinet as mn

# 1 构建多层网络
# 1.1 从文件读取数据并构建多层网络
g1 = mn.build_network("Krackhardt-High-Tech", "./")
print(g1)
# g1 = mn.build_network2("Krackhardt-High-Tech", "./")
# 1.2 生成随机多层网络
# g2 = mn.generate_simulation_network(1, 5, 1.1, 0)

# # 2 查看网络属性
# # 2.1 查看节点属性
# print(g1.node)
# # 2.2 查看层属性
# print(g1.layer)
# # 2.3 查看边连接情况
# print(g1.edge)

# 3 生成布局
# 3.1 多层布局算法：independent_layout/same_layout/wang_layout/random_layout
layout1 = mn.independent_layout(g1)
# 3.2 单层布局算法：kamada_kawai_layout/random_layout
# layout2 = mn.kamada_kawai_layout(g2)
print(layout1)
# 4 绘制
# 4.1 绘制多层网络：draw_multilayer_network
# mn.draw_multilayer_network(g1, layout=layout1)
# 4.2 绘制单层网络：draw_single_layer_networkx/draw_multilayer_network
# mn.draw_multilayer_network(g2, layout=layout2)

# eel.init('web')
# eel.start('dashboard')
