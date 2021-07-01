#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Lang Huang, Rainbowsecret
# Some methods used by main methods.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import torchcontrib
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

from lib.utils.tools.logger import Logger as Log


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class OptimScheduler(object):
    def __init__(self, configer):
        self.configer = configer

    def init_optimizer(self, net_params):
        optimizer = None
        if self.configer.get('optim', 'optim_method') == 'sgd':
            optimizer = SGD(net_params,
                            lr=self.configer.get('lr', 'base_lr'),
                            momentum=self.configer.get('optim', 'sgd')['momentum'],
                            weight_decay=self.configer.get('optim', 'sgd')['weight_decay'],
                            nesterov=self.configer.get('optim', 'sgd')['nesterov'])

        elif self.configer.get('optim', 'optim_method') == 'adam':
            optimizer = Adam(net_params,
                             lr=self.configer.get('lr', 'base_lr'),
                             betas=self.configer.get('optim', 'adam')['betas'],
                             eps=self.configer.get('optim', 'adam')['eps'],
                             weight_decay=self.configer.get('optim', 'adam')['weight_decay'])
        elif self.configer.get('optim', 'optim_method') == 'adamw':
            optimizer = AdamW(net_params,
                              lr=self.configer.get('lr', 'base_lr'),
                              betas=self.configer.get('optim', 'adamw')['betas'],
                              eps=self.configer.get('optim', 'adamw')['eps'],
                              weight_decay=self.configer.get('optim', 'adamw')['weight_decay'])

        else:
            Log.error('Optimizer {} is not valid.'.format(self.configer.get('optim', 'optim_method')))
            exit(1)

        policy = self.configer.get('lr', 'lr_policy')

        scheduler = None
        if policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer,
                                            self.configer.get('lr', 'step')['step_size'],
                                            gamma=self.configer.get('lr', 'step')['gamma'])

        elif policy == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 self.configer.get('lr', 'multistep')['stepvalue'],
                                                 gamma=self.configer.get('lr', 'multistep')['gamma'])

        elif policy == 'lambda_poly':
            if os.environ.get('lambda_poly_power'):
                _lambda_poly_power = float(os.environ.get('lambda_poly_power'))
                Log.info('Use lambda_poly policy with power {}'.format(_lambda_poly_power))
                lambda_poly = lambda iters: pow((1.0 - iters / self.configer.get('solver', 'max_iters')),
                                                _lambda_poly_power)
            elif self.configer.exists('lr', 'lambda_poly'):
                Log.info('Use lambda_poly policy with power {}'.format(self.configer.get('lr', 'lambda_poly')['power']))
                lambda_poly = lambda iters: pow((1.0 - iters / self.configer.get('solver', 'max_iters')),
                                                self.configer.get('lr', 'lambda_poly')['power'])
            else:
                Log.info('Use lambda_poly policy with default power 0.9')
                lambda_poly = lambda iters: pow((1.0 - iters / self.configer.get('solver', 'max_iters')), 0.9)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)

        elif policy == 'lambda_cosine':
            lambda_cosine = lambda iters: (math.cos(math.pi * iters / self.configer.get('solver', 'max_iters'))
                                           + 1.0) / 2
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cosine)

        elif policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode=self.configer.get('lr', 'plateau')['mode'],
                                                       factor=self.configer.get('lr', 'plateau')['factor'],
                                                       patience=self.configer.get('lr', 'plateau')['patience'],
                                                       threshold=self.configer.get('lr', 'plateau')['threshold'],
                                                       threshold_mode=self.configer.get('lr', 'plateau')['thre_mode'],
                                                       cooldown=self.configer.get('lr', 'plateau')['cooldown'],
                                                       min_lr=self.configer.get('lr', 'plateau')['min_lr'],
                                                       eps=self.configer.get('lr', 'plateau')['eps'])

        elif policy == 'swa_lambda_poly':
            optimizer = torchcontrib.optim.SWA(optimizer)
            normal_max_iters = int(self.configer.get('solver', 'max_iters') * 0.75)
            swa_step_max_iters = (self.configer.get('solver',
                                                    'max_iters') - normal_max_iters) // 5 + 1  # we use 5 ensembles here

            def swa_lambda_poly(iters):
                if iters < normal_max_iters:
                    return pow(1.0 - iters / normal_max_iters, 0.9)
                else:  # set lr to half of initial lr and start swa
                    return 0.5 * pow(1.0 - ((iters - normal_max_iters) % swa_step_max_iters) / swa_step_max_iters, 0.9)

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=swa_lambda_poly)

        elif policy == 'swa_lambda_cosine':
            optimizer = torchcontrib.optim.SWA(optimizer)
            normal_max_iters = int(self.configer.get('solver', 'max_iters') * 0.75)
            swa_step_max_iters = (self.configer.get('solver',
                                                    'max_iters') - normal_max_iters) // 5 + 1  # we use 5 ensembles here

            def swa_lambda_cosine(iters):
                if iters < normal_max_iters:
                    return (math.cos(math.pi * iters / normal_max_iters) + 1.0) / 2
                else:  # set lr to half of initial lr and start swa
                    return 0.5 * (math.cos(
                        math.pi * ((iters - normal_max_iters) % swa_step_max_iters) / swa_step_max_iters) + 1.0) / 2

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=swa_lambda_cosine)

        elif policy == 'warmup_cosine':
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=1000,
                                             t_total=self.configer.get('solver', 'max_iters'))

        else:
            Log.error('Policy:{} is not valid.'.format(policy))
            exit(1)

        return optimizer, scheduler

    def update_optimizer(self, net, optim_method, lr_policy):
        self.configer.update(('optim', 'optim_method'), optim_method)
        self.configer.update(('lr', 'lr_policy'), lr_policy)
        optimizer, scheduler = self.init_optimizer(net)
        return optimizer, scheduler
