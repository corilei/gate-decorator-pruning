#!/usr/bin/python
# -*- coding: utf-8 -*-
# Description: 
# Created: lei.cheng 2022/4/3
# Modified: lei.cheng 2022/4/3
''' setting before run. every notebook should include this code. '''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys

_r = os.getcwd().split('/')
_p = '/'.join(_r[:_r.index('gate-decorator-pruning')+1])
print('Change dir from %s to %s' % (os.getcwd(), _p))
os.chdir(_p)
sys.path.append(_p)

from config import parse_from_dict
parse_from_dict({
    "base": {
        "task_name": "resnet56_cifar10_ticktock",
        "cuda": True,
        "seed": 0,
        "checkpoint_path": "",
        "epoch": 0,
        "multi_gpus": True,
        "fp16": False
    },
    "model": {
        "name": "cifar.resnet56",
        "num_class": 10,
        "pretrained": False
    },
    "train": {
        "trainer": "normal",
        "max_epoch": 160,
        "optim": "sgd",
        "steplr": [
            [80, 0.1],
            [120, 0.01],
            [160, 0.001]
        ],
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "nesterov": False
    },
    "data": {
        "type": "cifar10",
        "shuffle": True,
        "batch_size": 128,
        "test_batch_size": 128,
        "num_workers": 4
    },
    "loss": {
        "criterion": "softmax"
    },
    "gbn": {
        "sparse_lambda": 1e-3,
        "flops_eta": 0,
        "lr_min": 1e-3,
        "lr_max": 1e-2,
        "tock_epoch": 10,
        "T": 10,
        "p": 0.002
    }
})
from config import cfg

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from logger import logger
from main import set_seeds, recover_pack, adjust_learning_rate, _step_lr, _sgdr
from models import get_model
from utils import dotdict

from prune.universal import Meltable, GatedBatchNorm2d, Conv2dObserver, IterRecoverFramework, FinalLinearObserver
from prune.utils import analyse_model, finetune

set_seeds()
pack = recover_pack()

model_dict = torch.load('./ckps/resnet56_cifair10_baseline.ckp', map_location='cpu' if not cfg.base.cuda else 'cuda')
pack.net.module.load_state_dict(model_dict)

GBNs = GatedBatchNorm2d.transform(pack.net)
for gbn in GBNs:
    gbn.extract_from_bn()

pack.optimizer = optim.SGD(
    pack.net.parameters() ,
    lr=2e-3,
    momentum=cfg.train.momentum,
    weight_decay=cfg.train.weight_decay,
    nesterov=cfg.train.nesterov
)

import uuid

def bottleneck_set_group(net):
    layers = [
        net.module.layer1,
        net.module.layer2,
        net.module.layer3
    ]
    for m in layers:
        masks = []
        if m == net.module.layer1:
            masks.append(pack.net.module.bn1)
        for mm in m.modules():
            if mm.__class__.__name__ == 'BasicBlock':
                if len(mm.shortcut._modules) > 0:
                    masks.append(mm.shortcut._modules['1'])
                masks.append(mm.bn2)

        group_id = uuid.uuid1()
        for mk in masks:
            mk.set_groupid(group_id)

bottleneck_set_group(pack.net)

def clone_model(net):
    model = get_model()
    gbns = GatedBatchNorm2d.transform(model.module)
    model.load_state_dict(net.state_dict())
    return model, gbns

cloned, _ = clone_model(pack.net)
BASE_FLOPS, BASE_PARAM = analyse_model(cloned.module, torch.randn(1, 3, 32, 32).cuda())
print('%.3f MFLOPS' % (BASE_FLOPS / 1e6))
print('%.3f M' % (BASE_PARAM / 1e6))
del cloned


def eval_prune(pack):
    cloned, _ = clone_model(pack.net)
    _ = Conv2dObserver.transform(cloned.module)
    cloned.module.linear = FinalLinearObserver(cloned.module.linear)
    cloned_pack = dotdict(pack.copy())
    cloned_pack.net = cloned
    Meltable.observe(cloned_pack, 0.001)
    Meltable.melt_all(cloned_pack.net)
    flops, params = analyse_model(cloned_pack.net.module, torch.randn(1, 3, 32, 32).cuda())
    del cloned
    del cloned_pack

    return flops, params

pack.trainer.test(pack)

pack.tick_trainset = pack.train_loader
prune_agent = IterRecoverFramework(pack, GBNs, sparse_lambda = cfg.gbn.sparse_lambda, flops_eta = cfg.gbn.flops_eta, minium_filter = 3)

LOGS = []
flops_save_points = set([40, 38, 35, 32, 30])

iter_idx = 0
prune_agent.tock(lr_min=cfg.gbn.lr_min, lr_max=cfg.gbn.lr_max, tock_epoch=cfg.gbn.tock_epoch)
while True:
    left_filter = prune_agent.total_filters - prune_agent.pruned_filters
    num_to_prune = int(left_filter * cfg.gbn.p)
    info = prune_agent.prune(num_to_prune, tick=True, lr=cfg.gbn.lr_min)
    flops, params = eval_prune(pack)
    info.update({
        'flops': '[%.2f%%] %.3f MFLOPS' % (flops / BASE_FLOPS * 100, flops / 1e6),
        'param': '[%.2f%%] %.3f M' % (params / BASE_PARAM * 100, params / 1e6)
    })
    LOGS.append(info)
    print(
        'Iter: %d,\t FLOPS: %s,\t Param: %s,\t Left: %d,\t Pruned Ratio: %.2f %%,\t Train Loss: %.4f,\t Test Acc: %.2f' %
        (iter_idx, info['flops'], info['param'], info['left'], info['total_pruned_ratio'] * 100, info['train_loss'],
         info['after_prune_test_acc']))

    iter_idx += 1
    if iter_idx % cfg.gbn.T == 0:
        print('Tocking:')
        prune_agent.tock(lr_min=cfg.gbn.lr_min, lr_max=cfg.gbn.lr_max, tock_epoch=cfg.gbn.tock_epoch)

    flops_ratio = flops / BASE_FLOPS * 100
    for point in [i for i in list(flops_save_points)]:
        if flops_ratio <= point:
            torch.save(pack.net.module.state_dict(), './logs/resnet56_cifar10_ticktock/%s.ckp' % str(point))
            flops_save_points.remove(point)

    if len(flops_save_points) == 0:
        break



