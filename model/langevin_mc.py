import logging

import numpy as np
import torch

from utils.graph_utils import check_adjs_symmetry, mask_adjs


def func_x(x):
    # return torch.sigmoid((x - 0.5))
    return x
    # return 0/.25 * torch.log((x + 1e-5) / (1. - x + 1e-5)) + 0.5


class LangevinMCSampler(object):
    def __init__(self, dev, eps=0.3,
                 grad_step_size=1.0,  # lambda in langevin dynamics
                 step_num=100,
                 max_node_num=None,
                 fixed_node_number=True,
                 **kwargs):
        super().__init__()
        self.dev = dev
        self.step_num = step_num
        self.grad_step_size = grad_step_size
        self.sigma = np.sqrt(self.grad_step_size * 2) * eps
        self.eps = eps
        self.ac_r_range = [0.5, 0.6]
        self.factor = 0.8
        self.max_node_num = max_node_num
        self.fixed_node_number = fixed_node_number

    def _update_step_size(self, grad_step_size):
        if grad_step_size is None:
            return
        self.grad_step_size = grad_step_size
        if isinstance(self.grad_step_size, torch.Tensor):
            self.sigma = (self.grad_step_size * 2).sqrt() * self.eps
        else:
            self.sigma = np.sqrt(self.grad_step_size * 2) * self.eps

    def gen_init_sample(self, batch_size, max_node_num, node_flags=None):
        adjs_c = torch.randn((batch_size, max_node_num, max_node_num),
                             dtype=torch.float32).triu(diagonal=1).abs().to(self.dev)
        adjs_c = (adjs_c + adjs_c.transpose(-1, -2))

        # adjs_c = torch.zeros([batch_size, max_node_num, max_node_num], dtype=torch.float32).to(self.dev)
        # flag_b = torch.ones([batch_size, node_number], dtype=torch.float32).to(dev)
        if node_flags is None:
            _, node_flags = self.adj_to_int(adjs_c)
        else:
            adjs_c = mask_adjs(adjs_c, node_flags)
        return adjs_c, node_flags

    @staticmethod
    def adj_to_int(adjs_c, to_int=False):
        if to_int:
            adjs_d = torch.where(adjs_c < 0.5, torch.zeros_like(adjs_c), torch.ones_like(adjs_c))
        else:
            adjs_d = func_x(adjs_c)

        # adjs_d = add_self_loop_if_not_exists(adjs_d)
        # node_flags = torch.ones([adjs_d.size(0), adjs_d.size(1)]).to(adjs_d)
        node_flags = adjs_d.sum(-1).squeeze(-1).gt(1e-5).to(torch.float32)
        return adjs_d, node_flags

    def end_sample(self, adjs, node_flags=None, to_int=True):
        return self.adj_to_int(adjs.detach(), to_int=to_int)

    def sample(self, batch_size, score_func,
               max_node_num=None, step_num=None,
               step_size=None,
               init_adjs=None, init_flags=None, is_final=True,
               eps=None):
        if eps is not None:
            self.eps = eps
        # note that `energy_model_prob_func` is exp(energy_func)
        if step_size is not None:
            self._update_step_size(step_size)
        if max_node_num is None:
            max_node_num = self.max_node_num
        if step_num is None:
            if isinstance(self.step_num, int):
                step_num = self.step_num
            elif isinstance(self.step_num, list) or isinstance(self.step_num, tuple):
                step_num = np.random.randint(min(self.step_num), 1 + max(self.step_num))

        if init_adjs is not None and init_flags is not None:
            adjs, node_flags = init_adjs, init_flags
        else:
            assert False
            # adjs, node_flags = self.gen_init_sample(batch_size, max_node_num)
        check_adjs_symmetry(adjs)
        logging.debug('-' * 60)
        for step in range(step_num):
            take_log = (step % 1 == 0)
            adjs, node_flags = self._step_sample(step, score_func, adjs, node_flags, log=take_log)
            check_adjs_symmetry(adjs)
        adjs.detach_()
        node_flags.detach_()
        check_adjs_symmetry(adjs)
        if is_final:
            adjs, node_flags = self.end_sample(adjs, node_flags)
        return adjs, node_flags

    def _add_sym_normal_noise(self, to_add, sigma=None):
        if sigma is None:
            sigma = self.sigma
        noise = torch.randn_like(to_add).triu(1) * sigma
        noise_s = noise + noise.transpose(-1, -2)
        check_adjs_symmetry(noise_s)
        check_adjs_symmetry(to_add)
        return noise_s + to_add

    def _step_sample(self, step, score_func, adjs, node_flags, log=True):
        adjs_c = adjs
        adjs_c = self._add_sym_normal_noise(adjs_c)
        adjs_c = mask_adjs(adjs_c, node_flags)
        check_adjs_symmetry(adjs_c)

        score = score_func(adjs_c, node_flags)
        # print((score - score.mean(0)).abs().sum().detach().cpu().item())
        # print((score.std(0).mean()).detach().cpu().item())
        # print(np.array2string(score.std().detach().cpu().numpy(), precision=2,
        #                                          separator='\t', prefix='\t'))
        check_adjs_symmetry(score)

        delta = self.grad_step_size * score
        new_adjs_c = adjs_c + delta

        adjs_c = new_adjs_c
        if log:
            logging.debug(f"LG MC: step {step:5d}\t|" +
                          "score: {:+.2e}\t|new_score_d: {:+.2e}\t"
                          "|adj_mean: {:.2e}\t|adj_std: {:.2e}\t|delta_abs_mean: {:+.2e}\t|"
                          .format(score.norm(dim=[-1, -2]).mean().item(),
                                  score_func(self.adj_to_int(adjs_c, to_int=False)[0],
                                             node_flags).norm(dim=[-1, -2]).mean().item(),
                                  adjs_c.mean([0, 1, 2]).item(),
                                  adjs_c.std([0, 1, 2]).item(),
                                  (delta.abs().sum([1, 2]) / node_flags.sum(-1)**2).mean().item())
                          )
        return adjs_c, node_flags
