# provide some indices to evaluate the graph layout of the multi-layer network.

import numpy as np

__all__ = [
    "calculate_SAD",
    "calculate_crossing",
    "calculate_TEL"
]


# 计算节点移动总距离（sum of absolute difference）所有点从初始位置起，经中间步骤，到最终位置，途中总的移动长度）
def calculate_SAD(g, g_layout, nn):
    ln = len(g)//nn
    if ln == 1:
        return 0
    delta = np.zeros([nn, ln-1, 2])
    for i in range(nn):
        for j in range(ln-1):
            delta[i, j] = g_layout[i+nn*(j+1)] - g_layout[i+nn*j]
    nodesep = np.linalg.norm(delta, axis=-1)
    sad = np.sum(nodesep)
    return sad


# 计算多层网络布局中边交叉的数量
def calculate_crossing(g, g_layout, nn):
    # 边交叉数量crossing num
    cn = 0
    edge_list = list(g.edges())
    for i in range(g.number_of_edges()):
        for j in range(i+1, g.number_of_edges()):
            if edge_list[i][0]//nn == edge_list[j][0]//nn and edge_list[i][0] != edge_list[j][0] and edge_list[i][0] !=\
                    edge_list[j][1] and edge_list[i][1] != edge_list[j][0] and edge_list[i][1] != edge_list[j][1]:
                i0j0 = g_layout.get(edge_list[j][0])-g_layout.get(edge_list[i][0])
                i0j1 = g_layout.get(edge_list[j][1])-g_layout.get(edge_list[i][0])
                i1j0 = g_layout.get(edge_list[j][0])-g_layout.get(edge_list[i][1])
                i1j1 = g_layout.get(edge_list[j][1])-g_layout.get(edge_list[i][1])
                j0i0 = -i0j0
                j0i1 = -i1j0
                j1i0 = -i0j1
                j1i1 = -i1j1
                if np.cross(i0j0, i0j1)*np.cross(i1j0, i1j1) < 0 and np.cross(j0i0, j0i1)*np.cross(j1i0, j1i1) < 0:
                    cn = cn + 1
    return cn


# 计算多层网络布局的总边长Total_Edge_Length
def calculate_TEL(g, g_layout, nn):
    # print(g_layout)
    tel = 0
    for pair_of_nodes in g.edges:
        u, v = pair_of_nodes
        tel = tel + np.linalg.norm(g_layout[u] - g_layout[v])
        # print(g_layout[u], g_layout[v])
    return tel