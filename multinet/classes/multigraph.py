# extend the Graph classes of NextworkX package, so that the new classes contain
# the unique fields of multi-layer network.


# 新的MultiGraph类定义，独立于networkx
class MultiGraph:
    # 构造函数，要求新建MultiGraph对象时必须提供层数和节点数
    def __init__(self, layer_num, node_num):
        # 属性node包含图中所有节点的基本信息
        # index和order属性其实可以去掉
        # self.node = [{'index': i, 'order': i} for i in range(node_num)]
        self.node = [{} for i in range(node_num)]
        # 属性edge包含图中所有边的基本信息，边以邻接表方式存储
        self.edge = [[dict() for _ in range(node_num)] for _ in range(layer_num)]
        # 属性layer_label包含每层网络的语义信息
        self.layer = [dict() for _ in range(layer_num)]

    # 添加边
    def add_edge(self, edge):
        layer_index, s_node_index, t_node_index, weight = edge
        self.edge[layer_index][s_node_index][t_node_index] = weight

    # 批量添加边
    def add_edge_from(self, edges):
        for edge in edges:
            self.add_edge(edge)

    # 根据给定的节点集返回子图
    def get_single_layer(self, layer_index):
        subgraph = MultiGraph(1, len(self.node))
        subgraph.edge = [self.edge[layer_index].copy()]
        return subgraph

    # 根据给定层计算层总边数，缺省则计算所有层的总边数和
    def number_of_edges(self, layer_index=None):
        if layer_index is None:
            layer = [i for i in range(len(self.layer))]
        else:
            layer = [layer_index]
        edge_num = 0
        for layer_index in layer:
            for s_node in self.edge[layer_index]:
                edge_num += len(s_node)
        return edge_num

    # 根据给定节点计算点度
    def degrees(self, node_index):
        degrees = 0
        for layer_index in range(len(self.layer)):
            degrees += len(self.edge[layer_index][node_index])
        return degrees

    # 根据给定节点（在此基础上也可以指定特定层）计算邻居
    def neighbors(self, node_index, layer_index=None):
        neighbors = []
        if layer_index is None:
            layer = [i for i in range(len(self.layer))]
        else:
            layer = [layer_index]
        for layer_index in layer:
            for t_node_index in self.edge[layer_index][node_index].keys():
                if t_node_index not in neighbors:
                    neighbors.append(t_node_index)
        return neighbors
