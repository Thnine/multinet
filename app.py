import numpy as np
from flask import Flask, request, jsonify
import random
from flask_cors import CORS
from matplotlib import pyplot as plt

import multinet as mn
import json
import numpy
import Simulated2 as sim
import path
app = Flask(__name__)
CORS(app, resources=r'/*')

resultformark = []


@app.route('/mesfilter/test', methods=['POST'])
def mesfilter_test():  # put application's code here
    print(request.args)
    username = request.args.get('username')
    password = request.args.get('password')
    return [username, password, "qwerqweqweqwe"]


@app.route('/mesfilter/mesSelectBytime', methods=['POST'])
def mesSelectBytime():  # put application's code here
    params = request.get_json()
    order = params["param"]["order"]
    print(order)
    # print(request.get_data())
    cotent = []
    if order == "1":
        for i in range(0, 100):
            data = {
                "source": "192.168.1." + str(random.randint(1, 200)),
                "target": "192.168.1." + str(random.randint(1, 200)),
                "weight": random.randint(1, 12),
            }
            cotent.append(data)
    elif order == "2":
        for i in range(0, 100):
            data = {
                "source": "192.168.1." + str(random.randint(1, 200)) + '-' + "192.168.1." + str(random.randint(1, 200)),
                "target": "192.168.1." + str(random.randint(1, 200)) + '-' + "192.168.1." + str(random.randint(1, 200)),
                "weight": random.randint(1, 12),
            }
            cotent.append(data)
    elif order == "3":
        for i in range(0, 100):
            data = {
                "source": "192.168.1." + str(random.randint(1, 200)) + '-' + "192.168.1." + str(
                    random.randint(1, 200)) + '-' + "192.168.1." + str(random.randint(1, 200)),
                "target": "192.168.1." + str(random.randint(1, 200)) + '-' + "192.168.1." + str(
                    random.randint(1, 200)) + '-' + "192.168.1." + str(random.randint(1, 200)),
                "weight": random.randint(1, 12),
            }
            cotent.append(data)
    elif order == "4":
        for i in range(0, 100):
            data = {
                "source": "192.168.1." + str(random.randint(1, 200)) + '-' + "192.168.1." + str(
                    random.randint(1, 200)) + '-' + "192.168.1." + str(
                    random.randint(1, 200)) + '-' + "192.168.1." + str(random.randint(1, 200)),
                "target": "192.168.1." + str(random.randint(1, 200)) + '-' + "192.168.1." + str(
                    random.randint(1, 200)) + '-' + "192.168.1." + str(
                    random.randint(1, 200)) + '-' + "192.168.1." + str(random.randint(1, 200)),
                "weight": random.randint(1, 12),
            }
            cotent.append(data)
    return cotent

def default(self, obj):
    if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
        numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
        numpy.uint16,numpy.uint32, numpy.uint64)):
        return int(obj)
    elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
        numpy.float64)):
        return float(obj)
    elif isinstance(obj, (numpy.ndarray,)): # add this line
        return obj.tolist() # add this line
    return json.JSONEncoder.default(self, obj)

@app.route('/mesfilter/mesDraw', methods=['POST'])
def mesDraw():
    params = request.get_json()
    node = params["node"]
    link = params["link"]
    lay = ['layerID layerLabel', '1 advice']
    g1 = mn.build_network2(lay,link,node)
    layout1 = mn.independent_layout(g1)
    res = []
    print(layout1)
    for key,value in layout1[0].items():
        data ={
            "x":value[0],
            'y':value[1]
        }
        res.append(data)
    print(res)
    return res

@app.route('/mesfilter/mesCross', methods=['POST'])
def mesCross():
    params = request.get_json()
    length_mat = params[0]
    all_path, all_ex = sim.sa(3000, pow(10, -1), 0.1, 200, length_mat,params[1])
    print(sim.search(all_path, length_mat), round(sim.e(sim.search(all_path, length_mat), length_mat)))
    iteration = len(all_path)
    res = sim.search(all_path, length_mat)
    # print(sim.e2(res,length_mat))
    all_path = np.array(all_path)
    all_ex = np.array(all_ex)
    # plt.xlabel("Iteration")
    # plt.ylabel("cross")
    # plt.plot(range(iteration), all_ex)
    # plt.show()

    return res

@app.route('/mesfilter/markov', methods=['POST'])
def mesMarkov():
    print("kaishi")
    data_dict=request.get_json()
    data = data_dict['data']
    parameter = data_dict['parameter']

    metrix = path.Tn_paths(data,parameter['k'],parameter['delta'])

    return metrix


## 王子潇布局调用的api
@app.route('/api/getWangZixiaoLayout',methods=['POST'])
def getWangZixiaoLayout():
    import multinet as mn
    reqParams = json.loads(request.get_data())
    layer_data = reqParams['layer_data']
    node_data = reqParams['node_data']
    edge_data = reqParams['edge_data']
    g1 = mn.build_network_pVersion(layer_data=layer_data,edge_data=edge_data,node_data=node_data)
    layout1 = mn.independent_layout(g1)
    
    ## 解决numpy数据无法进行json编码的问题
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32,
                                np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    return json.dumps(layout1,cls=NpEncoder)

## 获取联通子图
@app.route('/api/getConnectedSubGraph',methods=['POST'])
def getConnectedSubGraph():
    import networkx as nx
    reqParams = json.loads(request.get_data())
    nodes = reqParams['nodes'] #[id1,id2,...]
    links = reqParams['links'] #[[id3,id4],[id8,id4],...]
    to_find_node = reqParams['to_find_node'] #id5

    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    for l in links:
        G.add_edge(l[0],l[1])
    for v in nx.connected_components(G):
        if to_find_node in v:
            return json.dumps(list(v))
 


## test
@app.route('/api/test',methods=['POST'])
def test():
    reqParams = request.get_data()
    print(reqParams)
    return json.dumps(['hh'])

if __name__ == '__main__':
    app.run()
