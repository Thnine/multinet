__all__ = [
    "random_ordering",
    "cuthill_mcKee_ordering"
]


def bfs(G, source, sort_neighbors=None):
    from collections import deque
    import multinet as mn

    if sort_neighbors is None:
        sort_neighbors = mn.neighbors(G, source)
    bfs_list = [source]
    visited = {source}
    depth_limit = len(G.node)
    queue = deque([(source, depth_limit)])
    while len(queue) > 0:
        parent, depth_now = queue.popleft()
        for child in sort_neighbors(G, parent):
            if child not in visited:
                bfs_list.append(child)
                visited.add(child)
                queue.append((child, depth_now - 1))
    return bfs_list


def random_ordering(G):
    import random
    ordering = [_ for _ in range(len(G.node))]
    random.shuffle(ordering)
    return ordering


def _degree_incremental_neighbor(G, source):
    neighbors = G.neighbors(source)
    neighbors = sorted(neighbors, key=lambda x: G.degrees(x))
    return neighbors


def cuthill_mcKee_ordering(G):
    # 确定Cuthill-McKee算法的起点
    CM_ordering = []
    for i in range(len(G.node)):
        if i in CM_ordering:
            continue
        start_node_index, minimum_degree = i, G.degrees(i)
        for node_index in range(len(G.node)):
            if node_index not in CM_ordering and G.degrees(node_index) < minimum_degree:
                start_node_index, minimum_degree = node_index, G.degrees(node_index)
        CM_ordering.extend(bfs(G, start_node_index, sort_neighbors=_degree_incremental_neighbor))
    # print("Cuthill-McKee's starting point is: " + str(starting_p))
    return CM_ordering
