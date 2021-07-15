#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   group_sampler.py
Author:   zhanghao55@baidu.com
Date  :   21/01/16 17:12:45
Desc  :   
"""

import sys
import logging
import math
import torch
from torch.utils.data import Sampler


class GroupSampler(Sampler):
    """按组随机的分布式sampler，随机的时候，按顺序和指定组的大小将数据分为若干组
       组间随机打乱
       注意: batch_size应该是group_size的整数倍
    """

    def __init__(self, dataset, group_size=2, shuffle=True, seed=0):
        #  数据集大小一定被组大小整除
        self.dataset = dataset
        self.group_size = group_size

        self.num_samples =  len(self.dataset)
        assert self.num_samples % group_size == 0, "num_samples({}) is not devisible by group_size({})".format(self.num_samples, group_size)
        # 得到组数
        self.group_num_total = int(self.num_samples / group_size)
        logging.debug("group_num_total: {}".format(self.group_num_total))

        # 当前epoch
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            group_idxs = torch.randperm(self.group_num_total, generator=g).tolist()
        else:
            group_idxs = list(range(self.group_num_total))
        logging.debug("group_idxs: {}".format(group_idxs))

        ds_idxs = list()
        for group_id in group_idxs:
            for offset in range(self.group_size):
                ds_idxs.append(self.group_size * group_id + offset)

        # subsample
        assert len(ds_idxs) == self.num_samples

        logging.debug("ds_idx: {}".format(ds_idxs))
        return iter(ds_idxs)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
