# build multi-layer network model.


# 读取相对路径dir下名为ds的数据，并将ds的层、边、节点数据处理成一条一条的数据项
# 该函数对外部不可见
def _load_file(datasource: str, directory: str = ""):
    # 读取层数据文件
    layers_file = open(directory + datasource + '_layers.txt', 'r')
    # 读取边数据文件
    multiplex_file = open(directory + datasource + '_multiplex.edges', 'r')
    # 读取节点数据文件
    nodes_file = open(directory + datasource + '_nodes.txt', 'r')
    # 分割文本数据项
    layer_data = layers_file.read().split('\n')
    edge_data = multiplex_file.read().split('\n')
    node_data = nodes_file.read().split('\n')
    # 特殊处理：如果文本数据最后一行多出一个回车，则删除回车行的记录
    if layer_data[-1] == '':
        del layer_data[-1]
    if edge_data[-1] == '':
        del edge_data[-1]
    if node_data[-1] == '':
        del node_data[-1]
    #  关闭文件
    layers_file.close()
    multiplex_file.close()
    nodes_file.close()
    return layer_data, edge_data, node_data

# 搭建网络
def build_network2(layer_data, edge_data, node_data):

    import multinet as mn
    # ld为层数据字符串，mxd为边数据字符串，nd为点数据字符串
    # layer_data, edge_data, node_data = _load_file(datasource, directory)
    G = mn.MultiGraph(len(layer_data) - 1, len(node_data))
    # 添加层数据
    layer_attribute = layer_data[0].split(' ')
    for layer_index, data in enumerate(layer_data[1:]):
        data = data.split(' ')
        for i, attribute in enumerate(layer_attribute):
            G.layer[layer_index][attribute] = data[i]
    # 添加点数据
    node_attribute = ['nodeID', 'nodeAge', 'nodeTenure', 'nodeLevel', 'nodeDepartment']
    for node_index, data in enumerate(node_data):
        data = data.split(' ')
        for i, attribute in enumerate(node_attribute):
            G.node[node_index][attribute] = data[i]


    # 添加边数据
    for edge in edge_data:
        edge = edge.split(' ')
        #TODO:-1
        layer_index, s_node_index, t_node_index, weight = int(edge[0])-1 , int(edge[1]) , int(edge[2]), int(edge[3])
        G.add_edge((layer_index, s_node_index, t_node_index,1))
        G.add_edge((layer_index, t_node_index, s_node_index,1))

    # 返回图g
    return G
# 搭建网络
def build_network(datasource: str, directory: str = ""):
    """
    Parameters
    ----------
    datasource : str
        数据集名称
    directory : str
        数据集文件的相对路径
    """
    import multinet as mn
    # ld为层数据字符串，mxd为边数据字符串，nd为点数据字符串
    layer_data, edge_data, node_data = _load_file(datasource, directory)



    G = mn.MultiGraph(len(layer_data) - 1, len(node_data) - 1)
    # 添加层数据
    layer_attribute = layer_data[0].split(' ')
    # print(layer_attribute)
    for layer_index, data in enumerate(layer_data[1:]):
        data = data.split(' ')
        for i, attribute in enumerate(layer_attribute):
            G.layer[layer_index][attribute] = data[i]
    # 添加点数据
    node_attribute = node_data[0].split(' ')
    for node_index, data in enumerate(node_data[1:]):
        data = data.split(' ')
        for i, attribute in enumerate(node_attribute):
            G.node[node_index][attribute] = data[i]

    # 添加边数据
    for edge in edge_data:
        edge = edge.split(' ')
        #TODO:-1
        layer_index, s_node_index, t_node_index, weight = int(edge[0]) - 1, int(edge[1]) -1 , int(edge[2]) -1, int(edge[3])
        # print(layer_index, s_node_index, t_node_index, weight)
        G.add_edge((layer_index, s_node_index, t_node_index, weight))
        G.add_edge((layer_index, t_node_index, s_node_index, weight))

    # 返回图g
    return G

## 搭建网络,传参版
def build_network_pVersion(layer_data,edge_data,node_data):
    """
    Parameters
    ----------
    datasource : str
        数据集名称
    directory : str
        数据集文件的相对路径
    """
    import multinet as mn
    # ld为层数据字符串，mxd为边数据字符串，nd为点数据字符串 

    G = mn.MultiGraph(len(layer_data) - 1, len(node_data) - 1)
    # 添加层数据
    layer_attribute = layer_data[0].split(' ')
    # print(layer_attribute)
    for layer_index, data in enumerate(layer_data[1:]):
        data = data.split(' ')
        for i, attribute in enumerate(layer_attribute):
            G.layer[layer_index][attribute] = data[i]
    # 添加点数据
    node_attribute = node_data[0].split(' ')
    for node_index, data in enumerate(node_data[1:]):
        data = data.split(' ')
        for i, attribute in enumerate(node_attribute):
            G.node[node_index][attribute] = data[i]

    # 添加边数据
    for edge in edge_data:
        edge = edge.split(' ')
        #TODO:-1
        layer_index, s_node_index, t_node_index, weight = int(edge[0]) - 1, int(edge[1]) -1 , int(edge[2]) -1, int(edge[3])
        # print(layer_index, s_node_index, t_node_index, weight)
        G.add_edge((layer_index, s_node_index, t_node_index, weight))
        G.add_edge((layer_index, t_node_index, s_node_index, weight))

    # 返回图g
    return G


# 返回较小元素与较大元素构成的元组
def _swap(a, b):
    return (a, b) if a < b else (b, a)


# 生成随机多层网络数据
def generate_simulation_dataset_to_file(datasource, layer_num, node_num, edge_density, perturb, directory=""):
    # ds为数据源，ln为网络层数，nn为节点数，ld为边密度，perturb为扰动率
    # 打开数据存储文件
    import random
    from math import floor
    layers_file = open(directory + datasource + '_layers.txt', 'w')
    nodes_file = open(directory + datasource + '_nodes.txt', 'w')
    multiplex_file = open(directory + datasource + '_multiplex.edges', 'w')
    # 生成层标签信息
    layers_file.write('layerID layerLabel\n')
    for i in range(1, layer_num + 1):
        layers_file.write(str(i) + ' ' + str(i) + '\n')
    # 生成点标签信息
    nodes_file.write('nodeID nodeLabel\n')
    for i in range(1, node_num + 1):
        nodes_file.write(str(i) + ' ' + str(i) + '\n')
    # 单层边数edge_num等于点数乘以边密度
    en = floor(node_num * edge_density)
    # 生成初始的edges模板
    edges = []
    for i in range(en):
        while 1:
            while 1:
                s_node_index = random.randint(1, node_num)
                t_node_index = random.randint(1, node_num)
                if s_node_index != t_node_index:
                    break
            s_node_index, t_node_index = _swap(s_node_index, t_node_index)
            if [s_node_index, t_node_index] not in edges:
                edges.append([s_node_index, t_node_index])
                break
    perturb_edge_num = floor(en * perturb)
    total_edges = []
    for i in range(1, layer_num + 1):
        random.shuffle(edges)
        tmp_edges = edges
        for j in range(perturb_edge_num):
            s_node_index, t_node_index = edges[j]
            while ([s_node_index, t_node_index] in edges) or (s_node_index == t_node_index):
                s_node_index = random.randint(1, node_num)
                t_node_index = random.randint(1, node_num)
                s_node_index, t_node_index = _swap(s_node_index, t_node_index)
            tmp_edges[j] = [s_node_index, t_node_index]
        for j in range(en):
            total_edges.append([i, tmp_edges[j][0], tmp_edges[j][1]])
    for i in total_edges:
        multiplex_file.write(str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' 1' + '\n')
    layers_file.close()
    multiplex_file.close()
    nodes_file.close()


def generate_simulation_network(layer_num, node_num, edge_density, perturb):
    """
    Parameters
    ----------
    layer_num : int
        多层网络层数
    node_num : int
        多层网络节点数
    edge_density : float
        多层网络边密度
    perturb : float
        多层网络不同层的边变化率
    """
    import multinet as mn
    from math import floor
    import random
    # 构造MultiGraph对象
    G = mn.MultiGraph(layer_num, node_num)
    # 添加层标签
    for i in range(len(G.layer)):
        G.layer[i]["layerLabel"] = i
    # 构造边数据
    edge_num = floor(len(G.node) * edge_density)
    # 生成第0层的edges
    edges = []
    for i in range(edge_num):
        while 1:
            while 1:
                s_node_index = random.randint(0, len(G.node) - 1)
                t_node_index = random.randint(0, len(G.node) - 1)
                if s_node_index != t_node_index:
                    break
            s_node_index, t_node_index = _swap(s_node_index, t_node_index)
            if [s_node_index, t_node_index] not in edges:
                edges.append([s_node_index, t_node_index])
                break
    for s_node_index, t_node_index in edges:
        G.edge[0][s_node_index][t_node_index] = G.edge[0][t_node_index][s_node_index] = 1
    perturb_edge_num = floor(edge_num * perturb)
    for layer_index in range(1, len(G.layer)):
        random.shuffle(edges)
        for j in range(perturb_edge_num):
            s_node_index, t_node_index = edges[j]
            while ([s_node_index, t_node_index] in edges) or (s_node_index == t_node_index):
                s_node_index = random.randint(0, len(G.node) - 1)
                t_node_index = random.randint(0, len(G.node) - 1)
                s_node_index, t_node_index = _swap(s_node_index, t_node_index)
            edges[j] = [s_node_index, t_node_index]
        for s_node_index, t_node_index in edges:
            G.add_edge((layer_index, s_node_index, t_node_index, 1))
            G.add_edge((layer_index, t_node_index, s_node_index, 1))
    return G
