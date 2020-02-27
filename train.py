import logging
import time
import os

from easydict import EasyDict as edict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from evaluation.stats import eval_torch_batch

from model.langevin_mc import LangevinMCSampler
from sample import sample_main
from utils.arg_helper import edict2dict, parse_arguments, get_config, process_config, set_seed_and_logger, load_data
from utils.graph_utils import gen_list_of_data
from utils.loading_utils import get_mc_sampler, get_score_model, eval_sample_batch
from utils.visual_utils import plot_graphs_adj


def loss_func(score_list, grad_log_q_noise_list, sigma_list):
    loss = 0.0
    loss_items = []
    for score, grad_log_q_noise, sigma in zip(score_list, grad_log_q_noise_list, sigma_list):
        cur_loss = 0.5 * sigma ** 2 * ((score - grad_log_q_noise) ** 2).sum(dim=[-1, -2]).mean()
        loss_items.append(cur_loss.detach().cpu().item())
        loss = loss + cur_loss
    assert isinstance(loss, torch.Tensor)
    return loss, loss_items


def fit(model, optimizer, mcmc_sampler, train_dl, max_node_number, max_epoch=20, config=None,
        save_interval=50,
        sample_interval=1,
        sigma_list=None,
        sample_from_sigma_delta=0.0,
        test_dl=None
        ):
    logging.info(f"{sigma_list}, {sample_from_sigma_delta}")
    assert isinstance(mcmc_sampler, LangevinMCSampler)

    optimizer.zero_grad()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_dacey)
    for epoch in range(max_epoch):
        train_losses = []
        train_loss_items = []
        test_losses = []
        test_loss_items = []
        t_start = time.time()
        model.train()
        for train_adj_b, train_x_b in train_dl:
            train_adj_b = train_adj_b.to(config.dev)
            train_x_b = train_x_b.to(config.dev)
            train_node_flag_b = train_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
            if isinstance(sigma_list, float):
                sigma_list = [sigma_list]
            train_x_b, train_noise_adj_b, \
            train_node_flag_b, grad_log_q_noise_list = \
                gen_list_of_data(train_x_b, train_adj_b,
                                 train_node_flag_b, sigma_list)
            optimizer.zero_grad()
            # print('config.dev: ', config.dev)
            score = model(x=train_x_b,
                          adjs=train_noise_adj_b,
                          node_flags=train_node_flag_b)

            loss, loss_items = loss_func(score.chunk(len(sigma_list), dim=0), grad_log_q_noise_list, sigma_list)
            train_loss_items.append(loss_items)
            loss.backward()

            optimizer.step()
            train_losses.append(loss.detach().cpu().item())
        scheduler.step(epoch)
        assert isinstance(model, nn.Module)
        model.eval()

        for test_adj_b, test_x_b in test_dl:
            test_adj_b = test_adj_b.to(config.dev)
            test_x_b = test_x_b.to(config.dev)
            test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
            test_x_b, test_noise_adj_b, test_node_flag_b, grad_log_q_noise_list = \
                gen_list_of_data(test_x_b, test_adj_b,
                                 test_node_flag_b, sigma_list)
            with torch.no_grad():
                score = model(x=test_x_b, adjs=test_noise_adj_b,
                              node_flags=test_node_flag_b)
            loss, loss_items = loss_func(score.chunk(len(sigma_list), dim=0), grad_log_q_noise_list, sigma_list)
            test_loss_items.append(loss_items)
            test_losses.append(loss.detach().cpu().item())

        mean_train_loss = np.mean(train_losses)
        mean_test_loss = np.mean(test_losses)
        mean_train_loss_item = np.mean(train_loss_items, axis=0)
        mean_train_loss_item_str = np.array2string(mean_train_loss_item, precision=2, separator="\t", prefix="\t")
        mean_test_loss_item = np.mean(test_loss_items, axis=0)
        mean_test_loss_item_str = np.array2string(mean_test_loss_item, precision=2, separator="\t", prefix="\t")

        logging.info(f'epoch: {epoch:03d}| time: {time.time() - t_start:.1f}s| '
                     f'train loss: {mean_train_loss:+.3e} | '
                     f'test loss: {mean_test_loss:+.3e} | ')

        logging.info(f'epoch: {epoch:03d}| '
                     f'train loss i: {mean_train_loss_item_str} '
                     f'test loss i: {mean_test_loss_item_str} | ')

        if epoch % save_interval == save_interval - 1:
            to_save = {
                'model': model.state_dict(),
                'sigma_list': sigma_list,
                'config': edict2dict(config),
                'epoch': epoch,
                'train_loss': mean_train_loss,
                'test_loss': mean_test_loss,
                'train_loss_item': mean_train_loss_item,
                'test_loss_item': mean_test_loss_item,
            }
            torch.save(to_save, os.path.join(config.model_save_dir,
                                             f"{config.dataset.name}_{sigma_list}.pth"))
            # torch.save(to_save, os.path.join(config.save_dir, "model.pth"))

        if epoch % sample_interval == sample_interval - 1:
            model.eval()
            test_adj_b, test_x_b = test_dl.__iter__().__next__()
            test_adj_b = test_adj_b.to(config.dev)
            test_x_b = test_x_b.to(config.dev)
            if isinstance(config.mcmc.grad_step_size, (list, tuple)):
                grad_step_size = config.mcmc.grad_step_size[0]
            else:
                grad_step_size = config.mcmc.grad_step_size
            step_size = grad_step_size * \
                        torch.tensor(sigma_list).to(test_x_b) \
                            .repeat_interleave(test_adj_b.size(0), dim=0)[..., None, None] ** 2
            test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
            test_x_b, test_noise_adj_b, test_node_flag_b, grad_log_q_noise_list = \
                gen_list_of_data(test_x_b, test_adj_b,
                                 test_node_flag_b, sigma_list)
            init_adjs = test_noise_adj_b
            with torch.no_grad():
                sample_b, _ = mcmc_sampler.sample(config.sample.batch_size,
                                                  lambda x, y: model(test_x_b, x, y),
                                                  max_node_num=max_node_number, step_num=None,
                                                  init_adjs=init_adjs, init_flags=test_node_flag_b,
                                                  is_final=True,
                                                  step_size=step_size)
                sample_b_list = sample_b.chunk(len(sigma_list), dim=0)
                init_adjs_list = init_adjs.chunk(len(sigma_list), dim=0)
                for sigma, sample_b, init_adjs in zip(sigma_list, sample_b_list, init_adjs_list):
                    sample_from_sigma = sigma + sample_from_sigma_delta
                    eval_sample_batch(sample_b, mcmc_sampler.end_sample(test_adj_b, to_int=True)[0], init_adjs,
                                      config.save_dir, title=f'epoch_{epoch}_{sample_from_sigma}.pdf')

                    if init_adjs is not None:
                        plot_graphs_adj(mcmc_sampler.end_sample(init_adjs, to_int=True)[0],
                                        node_num=test_node_flag_b.sum(-1).cpu().numpy(),
                                        title=f'epoch_{epoch}_{sample_from_sigma}_init.pdf',
                                        save_dir=config.save_dir)
                    result_dict = eval_torch_batch(mcmc_sampler.end_sample(test_adj_b, to_int=True)[0],
                                                   sample_b, methods=None)
                    logging.info(f'MMD {epoch} {sample_from_sigma}: {result_dict}')


def train_main(config, args):
    set_seed_and_logger(config, args)
    train_dl, test_dl = load_data(config)
    mc_sampler = get_mc_sampler(config)

    model = get_score_model(config)
    param_strings = []
    max_string_len = 126
    for name, param in model.named_parameters():
        if param.requires_grad:
            line = '.' * max(0, max_string_len - len(name) - len(str(param.size())))
            param_strings.append(f"{name} {line} {param.size()}")
    param_string = '\n'.join(param_strings)
    logging.info(f"Parameters: \n{param_string}")
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Parameters Count: {total_params}, Trainable: {total_trainable_params}")
    optimizer = optim.Adam(model.parameters(),
                           lr=config.train.lr_init,
                           betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=config.train.weight_decay)
    fit(model, optimizer, mc_sampler, train_dl,
        max_node_number=config.dataset.max_node_num,
        max_epoch=config.train.max_epoch,
        config=config,
        save_interval=config.train.save_interval,
        sample_interval=config.train.sample_interval,
        sigma_list=config.train.sigmas,
        sample_from_sigma_delta=0.0,
        test_dl=test_dl
        )

    sample_main(config, args)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    args = parse_arguments('train_ego_small.yaml')
    ori_config_dict = get_config(args)
    config_dict = edict(ori_config_dict.copy())
    process_config(config_dict)
    print(config_dict)
    train_main(config_dict, args)
