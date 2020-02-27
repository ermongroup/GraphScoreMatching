import logging
import os

import torch
from scipy import stats

from model.gcn import GCN
from model.gin import GIN
from model.gnn import MAX_DEG_FEATURES
from model.langevin_mc import LangevinMCSampler
from model.edp_gnn import EdgeDensePredictionGraphScoreNetwork
from model.score_network import ScoreNetwork, MLPScoreNetwork, ConvScore, UNetResScore
from utils.visual_utils import plot_graphs_adj

NAME_TO_CLASS = {
    'gcn': GCN,
    'gin': GIN,
}


def gen_random_feature(batch_size, max_node_num, in_feature, dev, one_hot=False):
    if in_feature == 0:
        train_node_feature_b = None
    else:
        if one_hot:
            assert in_feature == max_node_num
            train_node_feature_b = torch.eye(max_node_num).unsqueeze(0).repeat([batch_size, 1, 1])
        else:
            train_node_feature_b = torch.randn(batch_size, max_node_num, in_feature) * 0.3
        train_node_feature_b = train_node_feature_b.to(dev)
    return train_node_feature_b


def get_norm_fit_stats(data, string=True):
    loc, scale = stats.norm.fit(data)
    n = stats.norm(loc=loc, scale=scale)
    kss, p = stats.kstest(data, n.cdf)
    if string:
        return '\t'.join([f'{v:.2f}' for v in [loc.item(), scale.item(), kss]]) + f'\t{p:.2e}'
    return loc.item(), scale.item(), kss, p


def get_mc_sampler(config):
    if config.mcmc.name == 'langevin':
        mc_sampler = LangevinMCSampler(eps=config.mcmc.eps
                                       if isinstance(config.mcmc.eps, float) else config.mcmc.eps[0],
                                       grad_step_size=config.mcmc.grad_step_size
                                       if isinstance(config.mcmc.grad_step_size, float)
                                       else config.mcmc.grad_step_size[0],
                                       max_node_num=config.dataset.max_node_num,
                                       step_num=config.mcmc.step_num,
                                       dev=config.dev)
    else:
        raise ValueError('Unknown mcmc method')
    return mc_sampler


def get_score_model(config, dev=None, **kwargs):
    if dev is None:
        dev = config.dev
    model_config = list(config.model.models.values())[0]
    in_features = config.dataset.in_feature
    feature_nums = [in_features + MAX_DEG_FEATURES] + model_config.feature_nums
    params = dict(model_config)
    params['feature_nums'] = feature_nums
    params.update(kwargs)
    if config.model.name == 'gnn':
        def gnn_model_func():
            return NAME_TO_CLASS[model_config.name](**params).to(dev)

        score_model = ScoreNetwork(gnn_module_func=gnn_model_func, feature_num=sum(feature_nums),
                                   stack_num=config.model.stack_num if 'stack_num' in config.model else 1).to(dev)
    elif config.model.name == 'mlp':
        score_model = MLPScoreNetwork(nef=64, max_node_number=config.dataset.max_node_num, dev=dev).to(dev)
    elif config.model.name == 'cov':
        score_model = ConvScore(nef=64, max_node_number=config.dataset.max_node_num, dev=dev).to(dev)
    elif config.model.name == 'unet':
        score_model = UNetResScore(nef=64, max_node_number=config.dataset.max_node_num, dev=dev).to(dev)
    elif config.model.name == 'edp-gnn':
        def gnn_model_func(**gnn_params):
            merged_params = params
            merged_params.update(gnn_params)
            return NAME_TO_CLASS[model_config.name](**merged_params).to(dev)

        feature_nums[0] = in_features
        score_model = EdgeDensePredictionGraphScoreNetwork(feature_num_list=feature_nums,
                                                           channel_num_list=model_config.channel_num_list,
                                                           max_node_number=config.dataset.max_node_num,
                                                           gnn_hidden_num_list=model_config.gnn_hidden_num_list,
                                                           gnn_module_func=gnn_model_func, dev=dev,
                                                           num_classes=len(config.train.sigmas)).to(dev)
    else:
        raise ValueError(f'Unknown model name {config.model.name}')
    logging.info('model: ' + str(score_model))
    return score_model


def eval_sample_batch(sample_b, test_adj_b, init_adjs, save_dir, title=""):
    delta = sample_b - test_adj_b
    init_delta = init_adjs - test_adj_b
    round_init_adjs = torch.where(init_adjs < 0.5, torch.zeros_like(init_adjs), torch.ones_like(init_adjs))
    round_init_delta = round_init_adjs - test_adj_b
    logging.info(f"sample delta_norm_mean: {delta.norm(dim=[1, 2]).mean().item():.3e} "
                 f"| init delta_norm_mean: {init_delta.norm(dim=[1, 2]).mean().item():.3e}"
                 f"| round init delta_norm_mean: {round_init_delta.norm(dim=[1, 2]).mean().item():.3e}")

    plot_graphs_adj(sample_b,
                    node_num=test_adj_b.sum(-1).gt(1e-5).sum(-1).cpu().numpy(),
                    title=title,
                    save_dir=save_dir)


def prepare_test_model(config):
    models = []
    if len(config.model_files) == 0:
        config.model_files = os.listdir(config.model_save_dir)
    print('config.model_files:', config.model_files)
    for file in config.model_files:
        ckp = torch.load(os.path.join(config.model_save_dir, file), map_location=config.dev)
        # if ckp['sigma_list'] not in config.train.sigmas:
        #     print(ckp['sigma_list'], config.train.sigmas)
        #     continue

        models.append((file, ckp['sigma_list'], [ckp, config.dev]))
    models.sort(key=lambda x: x[1], reverse=True)  # large sigma_list -> small sigma_list
    return models
