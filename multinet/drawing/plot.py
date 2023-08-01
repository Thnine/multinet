# plot the multi-layer network using graph or supra-adjacency matrix.

__all__ = [
    "draw_multilayer_network",
    "draw_single_layer_networkx",
    "draw_matrix"
]


def draw_single_layer_networkx(G, layout=None,**kwds):
    from inspect import signature
    import matplotlib.pyplot as plt
    import multinet as mn

    if mn.is_single_layer_network(G) is False:
        raise Exception(
            f"draw_single_layer_networkx函数适用于绘制单层网络，但所提供的G为{len(G.layer)}层网络。")

    # Get all valid keywords by inspecting the signatures of draw_networkx_nodes,
    # draw_networkx_edges, draw_networkx_labels

    valid_node_kwds = signature(draw_networkx_nodes).parameters.keys()
    valid_edge_kwds = signature(draw_networkx_edges).parameters.keys()
    valid_label_kwds = signature(draw_networkx_labels).parameters.keys()

    # Create a set with all valid keywords across the three functions and
    # remove the arguments of this function (draw_networkx)
    valid_kwds = (valid_node_kwds | valid_edge_kwds | valid_label_kwds) - {
        "G",
        "pos",
        "with_labels",
    }

    if any(k not in valid_kwds for k in kwds):
        invalid_args = ", ".join([k for k in kwds if k not in valid_kwds])
        raise ValueError(f"Received invalid argument(s): {invalid_args}")

    node_kwds = {k: v for k, v in kwds.items() if k in valid_node_kwds}
    edge_kwds = {k: v for k, v in kwds.items() if k in valid_edge_kwds}
    label_kwds = {k: v for k, v in kwds.items() if k in valid_label_kwds}

    if layout is None:
        layout = mn.kamada_kawai_layout(G)[0]

    draw_networkx_nodes(G, layout, **node_kwds)
    draw_networkx_edges(G, layout, **edge_kwds)
    draw_networkx_labels(G, layout, **label_kwds)

    plt.draw_if_interactive()


def draw_networkx_nodes(
        G,
        pos,
        nodelist=None,
        node_size=300,
        node_color="#1f78b4",
        node_shape="o",
        alpha=None,
        cmap=None,
        vmin=None,
        vmax=None,
        ax=None,
        linewidths=None,
        edgecolors=None,
        label=None
):
    import matplotlib as mpl
    import matplotlib.collections  # call as mpl.collections
    import matplotlib.pyplot as plt
    import numpy as np
    # 获取当前轴（若无轴则新建轴） Get the current Axes(gca)
    if ax is None:
        ax = plt.gca()

    if nodelist is None:
        nodelist = [i for i in range(len(G.node))]

    if len(nodelist) == 0:  # empty nodelist, no drawing
        return mpl.collections.PathCollection(None)

    xy = np.asarray([pos[v] for v in nodelist])

    node_collection = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_size,
        c=node_color,
        marker=node_shape,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidths=linewidths,
        edgecolors=edgecolors,
        label=label,
    )
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    node_collection.set_zorder(2)
    return node_collection


def draw_networkx_edges(
        G,
        pos,
        edgelist=None,
        width=1.0,
        edge_color="k",
        style="solid",
        alpha=None,
        edge_cmap=None,
        edge_vmin=None,
        edge_vmax=None,
        ax=None,
        label=None,
        nodelist=None
):
    from numbers import Number
    import matplotlib as mpl
    import matplotlib.collections  # call as mpl.collections
    import matplotlib.pyplot as plt
    import numpy as np

    # Some kwargs only apply to FancyArrowPatches. Warn users when they use
    # non-default values for these kwargs when LineCollection is being used
    # instead of silently ignoring the specified option

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list()
        for s_node_index in range(len(G.edge[0])):
            for t_node_index in G.edge[0][s_node_index].keys():
                edgelist.append((s_node_index, t_node_index))

    if len(edgelist) == 0:  # no edges!
        return []

    if nodelist is None:
        nodelist = [i for i in range(len(G.node))]

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    if (
            np.iterable(edge_color)
            and (len(edge_color) == len(edge_pos))
            and np.alltrue([isinstance(c, Number) for c in edge_color])
    ):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, mpl.colors.Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    def _draw_networkx_edges_line_collection():
        edge_collection = mpl.collections.LineCollection(
            edge_pos,
            colors=edge_color,
            linewidths=width,
            antialiaseds=(1,),
            linestyle=style,
            alpha=alpha,
        )
        edge_collection.set_cmap(edge_cmap)
        edge_collection.set_clim(edge_vmin, edge_vmax)
        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        return edge_collection

    # compute initial view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))
    w = maxx - minx
    h = maxy - miny

    # Draw the edges
    edge_viz_obj = _draw_networkx_edges_line_collection()

    # update view after drawing
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return edge_viz_obj


def draw_networkx_labels(
        G,
        pos,
        labels=None,
        font_size=12,
        font_color="k",
        font_family="sans-serif",
        font_weight="normal",
        alpha=None,
        bbox=None,
        horizontalalignment="center",
        verticalalignment="center",
        ax=None,
        clip_on=True,
):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = {n: n for n in range(len(G.node))}

    text_items = {}  # there is no text collection so we'll fake one
    for n, label in labels.items():
        (x, y) = pos[n]
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same
        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            transform=ax.transData,
            bbox=bbox,
            clip_on=clip_on,
        )
        text_items[n] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def draw_multilayer_network(G, layout=None):
    import multinet as mn
    import matplotlib.pyplot as plt

    if layout is None:
        layout = mn.independent_layout(G)
    # 画布fig，ax分割子图
    fig, ax = plt.subplots(1, len(G.layer))
    if len(G.layer) == 1:
        ax = [ax]
    # 绘制
    for layer_index in range(len(G.layer)):
        # 每绘制一个子层要拿到对应的子图
        subgraph = G.get_single_layer(layer_index)
        draw_single_layer_networkx(subgraph, ax=ax[layer_index],
                                   node_color='#666666',
                                   edge_color='#CCCCCC',
                                   font_color='#FFFFFF',
                                   node_size=200,
                                   font_size=9,
                                   layout=layout[layer_index])
        ax[layer_index].set_title('Relation: ' + str(G.layer[layer_index]["layerLabel"]))
    # 展示画布
    plt.show()


def draw_matrix(G, ordering=None, aggregated=False):
    import multinet as mn
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if aggregated is False:
        am = mn.get_supra_adjacency_matrix(G)
    else:
        am = mn.get_aggregated_network(G)

    if ordering is not None:
        am = _change_ordering(am, G, ordering)
    # print(type(am))
    # print(am)
    cmp = mpl.colors.ListedColormap(['white', 'black'])
    plt.matshow(am, cmap=cmp)
    plt.show()


# 根据给定节点排序ordering改变邻接矩阵排序
def _change_ordering(matrix, G, new_ordering):
    import numpy as np
    is_supra_matrix = True
    if len(G.node) == matrix.shape[0]:
        is_supra_matrix = False
    I = np.zeros(matrix.shape, dtype=int)
    for new_pos, key in enumerate(new_ordering):
        # 初始情况夏key就是order
        # old_pos = G.node[key]["order"]
        old_pos = key
        # 如果只是画图的话，就没必要真的修改原G的节点顺序
        # G.node[key]["ordering"] = new_pos
        if is_supra_matrix:
            for layer in range(len(G.layer)):
                offset = layer * len(G.node)
                # 这里我一直没搞懂为什么新编号放前，旧编号放后，而且变换后对角线元素会是错的，很怪
                I[new_pos + offset, old_pos + offset] = 1
        else:
            I[new_pos, old_pos] = 1
    matrix = I @ matrix @ I.T
    return matrix
