# calculate the layout for the given multi-layer network.

import random

__all__ = [
    "independent_layout",
    "same_layout",
    "multiforce_layout",
    "wang_layout",
    "kamada_kawai_layout",
    "random_layout"
]


def rescale_layout(pos, scale=1):
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos


def random_layout(G):
    import numpy as np
    pos_for_each_layer = list()
    for layer_index in range(len(G.layer)):
        pos = np.random.rand(len(G.node), 2)
        pos = pos.astype(np.float32)
        pos_for_each_layer.append(dict(zip([i for i in range(len(G.node))], pos)))
    return pos_for_each_layer


def kamada_kawai_layout(G):
    import numpy as np
    import multinet as mn

    if not mn.is_single_layer_network(G):
        raise Exception(f"Kamada-Kawai算法适用于计算单层网络布局，但所提供的G为{len(G.layer)}层网络。")

    dist = dict(mn.all_pairs_shortest_path_length(G))
    dist_mtx = np.zeros((len(G.node), len(G.node)))

    for s_node_index in range(len(G.node)):
        for t_node_index in range(len(G.node)):
            dist_mtx[s_node_index, t_node_index] = dist[s_node_index].get(t_node_index, 1e10)
            if dist_mtx[s_node_index, t_node_index] == 0:
                dist_mtx[s_node_index, t_node_index] = 1e10
    # print(dist_mtx)
    pos = random_layout(G)

    pos_arr = np.array([pos[0][i] for i in range(len(G.node))])
    pos = _kamada_kawai_solve(dist_mtx, pos_arr)
    pos = rescale_layout(pos, scale=1)
    return [dict(zip([i for i in range(len(G.node))], pos))]


def _kamada_kawai_solve(dist_mtx, pos_arr):
    # Anneal node locations based on the Kamada-Kawai cost-function,
    # using the supplied matrix of preferred inter-node distances,
    # and starting locations.

    import numpy as np
    import scipy as sp
    import scipy.optimize  # call as sp.optimize

    meanwt = 1e-3
    costargs = (np, 1 / (dist_mtx + np.eye(dist_mtx.shape[0]) * 1e-3), meanwt)

    optresult = sp.optimize.minimize(
        _kamada_kawai_costfn,
        pos_arr.ravel(),
        method="L-BFGS-B",
        args=costargs,
        jac=True,
    )

    return optresult.x.reshape((-1, 2))


def _kamada_kawai_costfn(pos_vec, np, invdist, meanweight):
    # Cost-function and gradient for Kamada-Kawai layout algorithm
    nNodes = invdist.shape[0]
    pos_arr = pos_vec.reshape((nNodes, 2))

    delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
    nodesep = np.linalg.norm(delta, axis=-1)
    direction = np.einsum("ijk,ij->ijk", delta, 1 / (nodesep + np.eye(nNodes) * 1e-3))

    offset = nodesep * invdist - 1.0
    offset[np.diag_indices(nNodes)] = 0

    cost = 0.5 * np.sum(offset ** 2)
    grad = np.einsum("ij,ij,ijk->ik", invdist, offset, direction) - np.einsum(
        "ij,ij,ijk->jk", invdist, offset, direction
    )

    # Additional parabolic term to encourage mean position to be near origin:
    sumpos = np.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * np.sum(sumpos ** 2)
    grad += meanweight * sumpos

    return (cost, grad.ravel())


# 计算独立布局
def independent_layout(G):
    layout = [{} for _ in range(len(G.layer))]
    for layer_index in range(len(G.layer)):
        subgraph = G.get_single_layer(layer_index)
        # 更新布局
        layout[layer_index].update(kamada_kawai_layout(subgraph)[0])
    return layout


# 计算相同布局
def same_layout(G):
    # 寻找边数最大的一层max_subgraph
    max_edge_num_layer = 0
    for layer_index in range(len(G.layer)):
        if G.number_of_edges(layer_index) > G.number_of_edges(max_edge_num_layer):
            max_edge_num_layer = layer_index
    subgraph = G.get_single_layer(max_edge_num_layer)
    # 计算max_subgraph的布局，并保存在tmp字典中
    single_layer_layout = kamada_kawai_layout(subgraph)[0]
    # 将max_subgraph层的布局扩展到所有层
    layout = [single_layer_layout for _ in range(len(G.layer))]
    return layout


# 计算multiforce布局
def multiforce_layout(g, ln, am, inla, interla, iterations=50):
    import numpy as np
    interla = interla / (ln - 1)
    # total_node_num，包含各层副本在内的总点数
    tnn = len(g)
    # node_num，层点数
    nn = tnn // ln
    k = (4 / nn) ** 0.5
    st = t = 2
    pos = random_layout(g)
    pos_arr = np.array([pos[n] for n in g])
    # pos_arr = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    # delta为ln*nn*nn矩阵，存储任意两点之间的横纵坐标之差，因此其中每个元素都是一个二元组（表征一个向量）。delta是一个反对称矩阵
    for iteration in range(iterations):
        delta = np.zeros((ln, nn, nn, 2))
        disp = np.zeros((tnn, 2))

        # calculate repulsive forces
        for i in range(tnn):
            for j in range(tnn):
                if i // nn == j // nn:
                    delta[i // nn, i % nn, j % nn] = pos_arr[i] - pos_arr[j]
        # print(delta)
        # deltanorm为ln*nn*nn矩阵，存储任意两点之间的欧式距离，因此deltanorm是一个对称矩阵
        deltanorm = np.linalg.norm(delta, axis=-1) + 1e-4
        # direction矩阵是delta矩阵‘算数除’以deltanorm矩阵（等于把delta标准化成模为1的单位向量了）
        direction = np.einsum("ijkl,ijk->ijkl", delta,
                              k ** 2 / deltanorm ** 2)
        # print(direction)
        disp = disp + np.sum(direction, axis=2).reshape((-1, 2))
        # print(disp)

        # calculate attractive forces inside each layer
        for i in range(ln):
            am[i] = np.triu(am[i])
        # print(am)
        direction = np.einsum("ijkl,ijk,ijk->ijkl", delta, deltanorm * inla / k, am)
        # print(direction)
        # print(-np.sum(direction, axis=2).reshape((-1, 2)) + np.sum(direction, axis=1).reshape((-1, 2)))
        disp = disp + (-np.sum(direction, axis=2).reshape((-1, 2)) + np.sum(direction, axis=1).reshape((-1, 2)))

        # calculate attractive forces across layers
        # print(pos_arr)
        for i in range(tnn):
            for j in range(i + 1, tnn):
                if i % nn == j % nn:
                    d = pos_arr[i] - pos_arr[j]
                    disp[i] = disp[i] - d * (d[0] ** 2 + d[1] ** 2) ** 0.5 / k * interla
                    disp[j] = disp[j] + d * (d[0] ** 2 + d[1] ** 2) ** 0.5 / k * interla

        # assign new positions
        # print(disp)
        dispnorm = np.linalg.norm(disp, axis=1) + 1e-4
        # print(dispnorm)
        for i in range(tnn):
            pos_arr[i] = pos_arr[i] + disp[i] / dispnorm[i] * min(dispnorm[i], t) + np.array(
                [random.random(), random.random()]) * 1e-4
            pos_arr[i][0] = min(1, max(-1, pos_arr[i][0]))
            pos_arr[i][1] = min(1, max(-1, pos_arr[i][1]))

        # print(pos_arr)
        # reduce the temperature
        t = t - st / iterations

    # 将pos按规模缩放
    # scale = 1
    # pos_arr = rescale_layout(pos_arr, scale=scale)

    return dict(zip(g, pos_arr))


# 构建wang_matrix
# 该函数还没有想好怎么改
def build_wang_matrix(am, nn, ln):
    import numpy as np
    wm = np.zeros((ln, ln, nn))
    for i in range(ln):
        for j in range(ln):
            for k in range(nn):
                wm[i, j, k] = np.sum(am[i, k]) / (np.sum(np.logical_or(am[i, k], am[j, k])) + 1e-10)
    return wm


# 计算wang布局
def wang_layout(G, weights=None, a=1, b=0.01, c=10):
    import numpy as np
    import multinet as mn

    if mn.is_single_layer_network(G) is True:
        raise Exception(f"我提出的算法适用于布局多层网络，但所提供的G为单层网络。")

    # 构建图G的最短路矩阵
    dist_mtx = np.zeros((len(G.layer), len(G.node), len(G.node)))
    for layer_index in range(len(G.layer)):
        subgraph = G.get_single_layer(layer_index)
        dist = dict(mn.all_pairs_shortest_path_length(subgraph))
        for s_node_index in range(len(G.node)):
            for t_node_index in range(len(G.node)):
                dist_mtx[layer_index, s_node_index, t_node_index] = dist[s_node_index].get(t_node_index, 1e10)
                if dist_mtx[layer_index, s_node_index, t_node_index] == 0:
                    dist_mtx[layer_index, s_node_index, t_node_index] = 1e10
    # 初始化点的位置
    pos = random_layout(G)
    # 把dict类型的pos数据转换为list类型，并存储在pos_arr中
    pos_arr = []
    for layer_index in range(len(G.layer)):
        for node_index in range(len(G.node)):
            pos_arr.append(pos[layer_index][node_index])
    pos_arr = np.array(pos_arr)
    # 把随机的pos转换成kk算法生成的pos
    pos = _wang_solve(dist_mtx, pos_arr, len(G.node), len(G.layer), weights, a, b, c)
    # 将pos按规模缩放
    pos = rescale_layout(pos, scale=1)

    final_pos = []
    for layer_index in range(len(G.layer)):
        pos_dict_per_layer = {}
        for node_index in range(len(G.node)):
            pos_dict_per_layer[node_index] = pos[layer_index * len(G.node) + node_index]
        final_pos.append(pos_dict_per_layer)

    return final_pos


# 算法迭代求解器
def _wang_solve(dist_mtx, pos_arr, nn, ln, weights, a, b, c):
    import numpy as np
    import scipy as sp
    import scipy.optimize  # call as sp.optimize
    # 损失参数costargs，包括numpy库，dist_mtx的倒数矩阵，层数layer_num
    tmp = np.eye(nn)
    costargs = (np, 1 / (dist_mtx + np.array([tmp for i in range(ln)]) * 1e-5), nn, ln, weights, a, b, c)
    # 优化结果optresult
    optresult = sp.optimize.minimize(
        # The objective function to be minimized.
        _wang_costfn,
        # Initial guess for L-BFGS-B, .ravel() puts pos_arr into 1-D data.
        pos_arr.ravel(),
        # Type of a solver.
        method="L-BFGS-B",
        # Extra arguments passed to the objective function and its derivatives (fun, jac and hess functions).
        args=costargs,
        # Method for computing the gradient vector.
        # If jac is a Boolean and is True,
        # fun is assumed to return a tuple (f, g) containing the objective function and the gradient.
        jac=True
    )
    return optresult.x.reshape((-1, 2))


# 算法损失函数
def _wang_costfn(pos_vec, np, invdist, nn, ln, weights, a, b, c):
    # Cost-function and gradient for Kamada-Kawai layout algorithm
    tnn = nn * ln
    pos_arr = pos_vec.reshape((-1, 2))
    # delta为ln*nn*nn矩阵，存储任意两点之间的横纵坐标之差，因此其中每个元素都是一个二元组（表征一个向量）。delta是一个反对称矩阵
    delta = np.zeros((ln, nn, nn, 2))
    for i in range(len(pos_arr)):
        for j in range(len(pos_arr)):
            if i // nn == j // nn:
                delta[i // nn, i % nn, j % nn] = pos_arr[j] - pos_arr[i]
    # nodesep为ln*nn*nn矩阵，存储任意两点之间的欧式距离，因此nodesep是一个对称矩阵
    nodesep = np.linalg.norm(delta, axis=-1)
    # direction矩阵是delta矩阵‘算数除’以nodesep矩阵（等于把delta标准化成模为1的单位向量了）
    tmp = np.eye(nn)
    direction = np.einsum("ijkl,ijk->ijkl", delta, 1 / (nodesep + np.array([tmp for i in range(ln)]) * 1e-5))
    # 用两点之间的欧式距离除以两点之间的最短路长度并减一，得到offset矩阵
    offset = nodesep * invdist - 1.0
    # offset矩阵对角线上元素归零
    for i in range(ln):
        tmp = offset[i]
        tmp[np.diag_indices(nn)] = 0
    # cost等于offset矩阵的上三角矩阵内所有元素的平方和
    cost = a * np.sum(offset ** 2)
    # 求梯度，这段真是绝了，很强
    # 前面是所有的出边，后面减去（取反）的是所有入边，这样就得出了一个点所有的指出向量
    for i in range(ln):
        tmp = 2 * a * np.einsum("ij,ij,ijk->ik", invdist[i], offset[i], direction[i]) - 2 * a * np.einsum(
            "ij,ij,ijk->jk", invdist[i], offset[i], direction[i]
        )
        if i == 0:
            grad = tmp
        else:
            grad = np.vstack((grad, tmp))
    grad = -grad

    # 点与点之间的斥力
    tmp = np.eye(nn)
    electronic_force = b / (nodesep + np.array([tmp for i in range(ln)]) * 1e10)
    # print(electronic_force)
    cost = cost + np.sum(electronic_force)
    for i in range(ln):
        tmp = np.einsum("ij,ijk->ik", electronic_force[i] ** 2, direction[i])
        if i == 0:
            tmp_grad = tmp
        else:
            tmp_grad = np.vstack((tmp_grad, tmp))
    # print(tmp_grad)
    grad = grad + tmp_grad

    # 点与壁之间的斥力
    # 我觉得这可能不是一个很好的想法，不如改成点点斥力在大于某一距离时消失，但是大于哪一距离呢？因为已经不存在画布大小的概念了
    # print(pos_arr)

    # Additional parabolic term to encourage mean position to be near origin:
    # 求各个点的copy族的平均位置
    manhatten = np.zeros(pos_arr.shape)
    for i in range(tnn):
        bi = i % nn
        if weights is None:
            manhatten[i] = np.sum([pos_arr[i] - pos_arr[bi + j * nn] for j in range(ln)], axis=0) / ln
        else:
            # for j in range(ln):
            # manhatten[i] = manhatten[i]+weights[j, i//nn, bi]*(pos_arr[i] - pos_arr[bi + j * nn])
            # manhatten[i] = manhatten[i] / np.sum([weights[j, i//nn, bi] for j in range(ln)])
            manhatten[i] = np.sum([weights[j, i // nn, bi] * (pos_arr[i] - pos_arr[bi + j * nn]) for j in range(ln)],
                                  axis=0) / (np.sum([weights[j, i // nn, bi] for j in range(ln)]) + 1e-10)
    # print(manhatten)
    euclidean = np.linalg.norm(manhatten, axis=-1)
    # print('cost:', cost, 10*np.sum(euclidean**2))
    cost += c * np.sum(euclidean ** 2)
    grad += 2 * c * (ln - 1) / ln * manhatten

    return (cost, grad.ravel())
