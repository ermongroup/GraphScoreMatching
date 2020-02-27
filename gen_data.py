import numpy as np

from utils.data_generators import gen_graph_list
from utils.visual_utils import plot_graphs_list

if __name__ == '__main__':
    file_name = 'community_small'
    res_graph_list = gen_graph_list(graph_type='community', possible_params_dict={
        'num_communities': [2],
        'max_nodes': np.arange(12, 21).tolist()
    }, corrupt_func=None, length=100, save_dir='data', file_name=file_name)
    plot_graphs_list(res_graph_list, title=file_name, save_dir='data')

    file_name = 'lobster_10_10_4k'
    res_graph_list = gen_graph_list(graph_type='lobster',
                                    possible_params_dict={
                                        'n': np.arange(5, 16).tolist(),
                                        'p1': [0.7],
                                        'p2': [0.5]
                                    }, corrupt_func=None, length=4096, save_dir='data', file_name=file_name,
                                    max_node=10,
                                    min_node=10)
    plot_graphs_list(res_graph_list, title=file_name, save_dir='data')
