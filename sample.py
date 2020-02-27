import logging
import pickle
import os

from easydict import EasyDict as edict
import numpy as np
import torch

from evaluation.stats import eval_torch_batch, adjs_to_graphs, eval_graph_list
from utils.arg_helper import mkdir, set_seed_and_logger, load_data, graphs_to_tensor, load_model, parse_arguments, \
    get_config
from utils.graph_utils import add_gaussian_noise
from utils.loading_utils import get_mc_sampler, eval_sample_batch, prepare_test_model
from utils.visual_utils import plot_graphs_list


def sample_main(ori_config, args):
    config = edict(ori_config.copy())
    config.save_dir = os.path.join(config.save_dir, 'sample')
    mkdir(config.save_dir)
    config.model_files = []
    config.init_sigma = 'inf'
    set_seed_and_logger(config, args)
    train_graph_list, test_graph_list = load_data(config, get_graph_list=True)
    mcmc_sampler = get_mc_sampler(config)
    models = prepare_test_model(config)
    max_node_number = config.dataset.max_node_num
    test_batch_size = config.test.batch_size

    def gen_init_data(batch_size):

        # sample node numbers and node features (if exist) from training set
        rand_idx = np.random.randint(0, len(train_graph_list), batch_size)
        graph_list = [train_graph_list[i] for i in rand_idx]
        base_adjs, base_x = graphs_to_tensor(config, graph_list)
        base_adjs, base_x = base_adjs.to(config.dev), base_x.to(config.dev)
        node_flags = base_adjs.sum(-1).gt(1e-5).to(dtype=torch.float32)  # mark isolate nodes as non-exist
        print(batch_size, node_flags.size(0), base_adjs.size(0), base_x.size(0), len(graph_list), rand_idx.shape)
        if isinstance(config.init_sigma, float):  # when generate from perturbed training data
            base_adjs, _ = add_gaussian_noise(base_adjs, node_flags=node_flags, sigma=config.init_sigma)
        else:  # when generate from noise
            base_adjs = mcmc_sampler.gen_init_sample(batch_size, max_node_number, node_flags=node_flags)[0]

        return base_adjs, base_x, node_flags

    assert len(models) == 1
    file, sigma_list, model_params = models[0]

    model = load_model(*model_params)
    sigma_list = sorted(sigma_list)
    if isinstance(config.mcmc.grad_step_size, (tuple, list)):
        step_size_list = config.mcmc.grad_step_size
    else:
        step_size_list = [config.mcmc.grad_step_size]
    if isinstance(config.mcmc.eps, (tuple, list)):
        eps_list = config.mcmc.eps
    else:
        eps_list = [config.mcmc.eps]
    print(step_size_list, eps_list)

    best_config = {
        'step_size_ratio': step_size_list[0],
        'eps': eps_list[0],
        'best_batch_mmd': {
            '': np.inf
        }
    }

    def run_sample(target_graph_list, step_size_ratio, eps, validation=True, eval_len=1024, methods=None):
        warm_up_count = 0
        gen_graph_list = []
        init_adjs, init_x, node_flags = gen_init_data(batch_size=test_batch_size * len(sigma_list))
        sample_x = init_x
        sample_node_flags = node_flags
        valid_adj_b_ori = graphs_to_tensor(config, target_graph_list)[0].to(config.dev)
        while len(gen_graph_list) < eval_len:
            step_size = step_size_ratio * \
                        torch.tensor(sigma_list).to(init_adjs).repeat_interleave(test_batch_size,
                                                                                 dim=0)[..., None, None] ** 2
            # print(f'ss: {config.mcmc.grad_step_size}, {sigma_list}, {step_size}')
            with torch.no_grad():
                sample_adjs, _ = mcmc_sampler.sample(config.sample.batch_size,
                                                     lambda x, y: model(
                                                         sample_x,
                                                         x, y),
                                                     max_node_num=max_node_number, step_num=None,
                                                     init_adjs=init_adjs, init_flags=sample_node_flags,
                                                     is_final=False,
                                                     step_size=step_size,
                                                     eps=eps)
            sample_adjs_list = sample_adjs.chunk(len(sigma_list), dim=0)
            if warm_up_count < len(sigma_list):
                if validation:
                    for ii in range(len(sigma_list)):
                        rounded_adjs, _ = mcmc_sampler.end_sample(sample_adjs_list[ii])
                        # logging.info(f'{file}, {sigma}, {step_size}')
                        # logging.debug('\n' + np.array2string(sample_b_list[0].detach()[0].cpu().numpy(), precision=2,
                        #                                      separator='\t', prefix='\t'))
                        pic_title = f'{file.split("/")[-1]}_{step_size_ratio}_{eps}_' \
                                    f'{warm_up_count}_{sigma_list[ii]}_sample.pdf'
                        eval_sample_batch(rounded_adjs, valid_adj_b_ori, init_adjs.chunk(len(sigma_list), dim=0)[ii],
                                          config.save_dir, title=pic_title)
                        result_dict = eval_torch_batch(valid_adj_b_ori, rounded_adjs, methods=methods)
                        logging.info(f'MMD {file} {warm_up_count} iter at {sigma_list[ii]}: {result_dict}')
                        logging.info(f'save fig at {config.save_dir} {pic_title}')
                warm_up_count += 1
            else:
                gen_graph_list.extend(adjs_to_graphs(mcmc_sampler.end_sample(sample_adjs_list[0]
                                                                             )[0].cpu().numpy()
                                                     )
                                      )
            new_init_adjs, new_init_x, new_node_flags = gen_init_data(batch_size=test_batch_size)
            init_adjs = torch.cat(list(sample_adjs_list[1:]) +
                                  [new_init_adjs], dim=0)
            sample_x = torch.cat([sample_x[sample_adjs_list[0].size(0):], new_init_x], dim=0)
            sample_node_flags = torch.cat([sample_node_flags[sample_adjs_list[0].size(0):], new_node_flags], dim=0)
        pic_title = f'{file.split("/")[-1]}_{step_size_ratio}_{eps}_{sigma_list}_final_sample.pdf'
        plot_graphs_list(graphs=gen_graph_list, title=pic_title, save_dir=config.save_dir)
        result_dict = eval_graph_list(test_graph_list, gen_graph_list, methods=methods)
        logging.info(f'MMD_full {file} {eval_len}: {result_dict}')
        return result_dict, gen_graph_list

    for step_size_ratio in step_size_list:
        for eps in eps_list:
            valid_result_dict, _ = run_sample(target_graph_list=train_graph_list[:config.test.batch_size],
                                              step_size_ratio=step_size_ratio,
                                              eps=eps,
                                              validation=False,
                                              eval_len=test_batch_size,
                                              methods=None)
            if np.sum(list(valid_result_dict.values())) < np.sum(list(best_config['best_batch_mmd'].values())):
                best_config = {
                    'step_size_ratio': step_size_ratio,
                    'eps': eps,
                    'best_batch_mmd': valid_result_dict
                }
    logging.info(f'best_config {file} iter: {best_config}')
    _, gen_graph_list = run_sample(target_graph_list=test_graph_list,
                                   step_size_ratio=best_config['step_size_ratio'],
                                   eps=best_config['eps'],
                                   validation=False,
                                   eval_len=1024)
    print(best_config)
    sample_dir = os.path.join(config.save_dir, 'sample_data')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    with open(os.path.join(sample_dir, file + f"_{best_config['step_size_ratio']}"
                                              f"_{best_config['eps']}_sample.pkl"), 'wb') as f:
        pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = parse_arguments('sample_com_small.yaml')
    config_dict = get_config(args)
    sample_main(config_dict, args)
