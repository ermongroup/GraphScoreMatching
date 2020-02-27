import os
import pickle

import networkx as nx
import numpy as np
import scipy.sparse as sp

from utils.arg_helper import mkdir
from utils.visual_utils import plot_graphs_list


# load ENZYMES and PROTEIN and DD dataset
def graph_load_batch(min_num_nodes=20, max_num_nodes=1000, name='ENZYMES', node_attributes=True, graph_labels=True):
    """
    load many graphs, e.g. enzymes
    :return: a list of graphs
    """
    print('Loading graph dataset: ' + str(name))
    G = nx.Graph()
    # load data
    path = 'dataset/' + name + '/'
    data_adj = np.loadtxt(path + name + '_A.txt', delimiter=',').astype(int)
    data_node_att = []
    if node_attributes:
        data_node_att = np.loadtxt(path + name + '_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path + name + '_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path + name + '_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path + name + '_graph_labels.txt', delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    print(G.number_of_nodes())
    print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if min_num_nodes <= G_sub.number_of_nodes() <= max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
    # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
    # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    print('Loaded')
    return graphs


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# load cora, citeseer and pubmed dataset
def graph_load(dataset='cora'):
    """
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    """
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pickle.load(open("dataset/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1')
        # print('loaded')
        objects.append(load)
        # print(load)
    # [x, tx, allx]: <class 'list'>: [(140, 1433), (1000, 1433), (1708, 1433)]
    # len(graph) == 2708
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    return features, G


def citeseer_ego(radius=3, node_min=50, node_max=400):
    _, G = graph_load(dataset='citeseer')
    G = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.convert_node_labels_to_integers(G)
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=radius)
        assert isinstance(G_ego, nx.Graph)
        if G_ego.number_of_nodes() >= node_min and (G_ego.number_of_nodes() <= node_max):
            G_ego.remove_edges_from(G_ego.selfloop_edges())
            graphs.append(G_ego)
    return graphs


def save_dataset(graphs, save_name):
    mkdir('data')
    file_path = os.path.join('data', save_name)
    print(save_name, len(graphs))
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(obj=graphs, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_path + '.txt', 'w') as f:
        f.write(save_name + '\n')
        f.write(str(len(graphs)))
    plot_graphs_list(graphs, title=save_name, save_dir='data')


if __name__ == '__main__':
    dataset_name = 'PROTEINS_full'
    suffix = '_30'
    graphs = graph_load_batch(min_num_nodes=20, max_num_nodes=30, name=dataset_name,
                              node_attributes=True, graph_labels=True)
    print(max([g.number_of_nodes() for g in graphs]))
    save_dataset(graphs, dataset_name + suffix)

    dataset_name = 'ego'
    suffix = '_small'
    graphs = citeseer_ego(radius=1, node_min=4, node_max=18)[:200]
    save_dataset(graphs, dataset_name+suffix)
    print(max([g.number_of_nodes() for g in graphs]))
