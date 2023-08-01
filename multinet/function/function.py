# 为MultiGraph类对象提供一些常用的工具方法
def is_single_layer_network(G):
    if len(G.layer) == 1:
        return True
    return False


# 社交网络图在求最短路时一般不考虑边权，所以这里直接用BFS求
def single_source_shortest_path_length(G, node_index):
    from collections import deque
    if not is_single_layer_network(G):
        raise Exception(
            f"single_source_shortest_path_length函数适用于计算单层网络的单源最短路径，但所提供的G为{len(G.layer)}层网络。")
    shortest_path_length = dict()
    visited = {node_index}
    que = deque([[node_index, 0]])
    while len(que) > 0:
        s_node_index, length = que.popleft()
        shortest_path_length[s_node_index] = length
        for t_node_index in G.edge[0][s_node_index].keys():
            if t_node_index not in visited:
                visited.add(t_node_index)
                que.append([t_node_index, length + 1])
    return shortest_path_length


def all_pairs_shortest_path_length(G):
    if not is_single_layer_network(G):
        raise Exception(
            f"all_pairs_shortest_path_length函数适用于计算单层网络的全点对间最短路径，但所提供的G为{len(G.layer)}层网络。")
    for node_index in range(len(G.node)):
        yield node_index, single_source_shortest_path_length(G, node_index)


def get_supra_adjacency_matrix(G):
    import numpy as np
    matrix = np.zeros((len(G.node) * len(G.layer), len(G.node) * len(G.layer)), dtype=int)
    for layer in range(len(G.layer)):
        offset = layer * len(G.node)
        for s in range(len(G.node)):
            for t in G.edge[layer][s].keys():
                matrix[s + offset, t + offset] = matrix[t + offset, s + offset] = 1
    return matrix


# 查找给定图中给定节点的所有邻居
def neighbors(G, node_index):
    return G.neighbors(node_index)


# 这个函数没写完
def adjacency_matrix(G):
    pass


def get_aggregated_network(G):
    import numpy as np
    return G
