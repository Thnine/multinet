import math
import random

import numpy as np
from matplotlib import pyplot as plt

mes = {
    1: [
        [1, 3, 1],
        [4, 2, 1],
        [5, 7, 1],
        [6, 1, 1],
        [7, 6, 1]
    ],
    2: [
        [1, 3, 2],
        [2, 4, 2],
        [3, 5, 2],
        [4, 1, 2],
        [5, 6, 2],
        [6, 7, 2],
        [7, 5, 2]
    ]
}

path_random = [1, 2, 3, 4, 5, 6, 7]


def initialization(path_random):
    return path_random


a = [1, 2, 3, 4, 5, 6, 7]


def generate_new(path):
    numbers = len(path)
    # 随机生成两个不重复的点
    positions = np.random.choice(list(range(numbers)), replace=False, size=2)
    lower_position = min(positions[0], positions[1])
    upper_position = max(positions[0], positions[1])
    # 将数列中段逆转
    mid_reversed = path[lower_position:upper_position + 1]
    mid_reversed.reverse()
    # 拼接生成新的数列
    new_path = path[:lower_position]
    new_path.extend(mid_reversed)
    new_path.extend(path[upper_position + 1:])
    return new_path


# print([7, 6, 5, 3, 4, 1, 2].index(1))

# length_mat为各个节点之间的最短距离矩阵
def e2(path, mes):
    loss = 0
    for key in mes:
        mesSingleTime = mes[key]
        for index in range(len(mesSingleTime)):
            for index2 in range(index + 1, len(mesSingleTime)):
                if (path.index(mesSingleTime[index][0]) < path.index(mesSingleTime[index2][0]) and path.index(
                        mesSingleTime[index][1]) > path.index(mesSingleTime[index2][1])):
                    # print(mesSingleTime[index],"jiaocha",mesSingleTime[index2])
                    loss += 1
                if (path.index(mesSingleTime[index][0]) > path.index(mesSingleTime[index2][0]) and path.index(
                        mesSingleTime[index][1]) < path.index(mesSingleTime[index2][1])):
                    # print(mesSingleTime[index], "jiaocha", mesSingleTime[index2])
                    loss += 1
        # print(mes[key])
    return loss

def e(path, mes):
    loss = 0
    for key in mes:
        mesSingleTime = mes[key]
        for index in range(len(mesSingleTime)):
            dis = abs(path.index(mesSingleTime[index][0]) - path.index(mesSingleTime[index][1]))
            loss += dis
        # print(mes[key])
    return loss

# print(e([2,4,1,3,5,7,6],mes))


def metropolis(e, new_e, t):
    if new_e <= e:
        return True
    else:
        p = math.exp((e - new_e) / t)
        return True if random.random() < p else False


def search(all_path, length_mat):
    best_e = 0xffff
    best_path = all_path[0]
    for path in all_path:
        ex = e(path, length_mat)
        if ex < best_e:
            best_e = ex
            best_path = path
    return best_path


# t0为初始温度，t_final为终止温度，alpha为冷却系数，inner_iter为内层迭代次数
def sa(t0, t_final, alpha, inner_iter, length_mat, ai):
    all_path = []
    all_ex = []
    init = initialization(ai)
    all_path.append(init)
    all_ex.append(e(init, length_mat))
    t = t0
    while t > t_final:
        path = search(all_path, length_mat)
        ex = e(path, length_mat)
        for i in range(inner_iter):
            new_path = generate_new(path)
            new_ex = e(new_path, length_mat)
            if metropolis(ex, new_ex, t):
                path = new_path
                ex = new_ex
        all_path.append(path)
        all_ex.append(ex)
        t = alpha * t
        # print("当前温度：", t)
    return all_path, all_ex



if __name__ == '__main__':
    length_mat = mes
    all_path, all_ex = sa(3000, pow(10, -1) , 0.98, 200, length_mat,a)
    print(search(all_path, length_mat), round(e(search(all_path, length_mat), length_mat)))
    iteration = len(all_path)
    all_path = np.array(all_path)
    all_ex = np.array(all_ex)
    plt.xlabel("Iteration")
    plt.ylabel("cross")
    plt.plot(range(iteration), all_ex)
    plt.show()
