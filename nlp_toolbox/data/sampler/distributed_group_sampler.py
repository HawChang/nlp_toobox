#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   distributed_group_sampler.py
Author:   zhanghao55@baidu.com
Date  :   21/01/16 17:12:45
Desc  :   
"""

import sys
import logging
import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist


class DistributedGroupSampler(Sampler):
    """��������ķֲ�ʽsampler�������ʱ�򣬰�˳���ָ����Ĵ�С�����ݷ�Ϊ������
       ����������
       ע��: batch_sizeӦ����group_size��������
    """

    def __init__(self, dataset, group_size=2, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        #  ���ݼ���Сһ�������С����
        self.dataset = dataset
        self.group_size = group_size

        dataset_size = len(self.dataset)
        assert dataset_size % group_size == 0, "dataset_size({}) is not devisible by group_size({})".format(dataset_size, group_size)
        # �õ�����
        self.group_num_total = int(dataset_size / group_size)
        logging.debug("group_num_total: {}".format(self.group_num_total))

        # ������
        self.num_replicas = num_replicas
        # ��ǰ����ID
        self.rank = rank
        # ��ǰepoch
        self.epoch = 0
        # group_num_each: ÿ��process_����䵽������
        group_num_each = int(math.ceil(self.group_num_total * 1.0 / num_replicas))
        # num_samples: һ��process�����䵽�������� ������*���С
        self.num_samples = group_num_each * group_size

        # pad_group_size Ϊ��ȫ����������batch�����ݶ���Ҫ���뵽������
        self.pad_group_size = group_num_each * self.num_replicas
        logging.debug("pad_group_size: {}".format(self.pad_group_size))
        # total_size Ϊ��ȫ����������batch�����ݶ���Ҫ���뵽�����ݼ���
        #self.total_size = self.num_samples * self.num_replicas
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


        # add extra samples to make it evenly divisible
        group_idxs += group_idxs[:(self.pad_group_size - len(group_idxs))]
        assert len(group_idxs) == self.pad_group_size

        # �õ���ǰrank���ֵ�����
        group_idxs = group_idxs[self.rank:self.pad_group_size:self.num_replicas]
        logging.debug("group_idxs: {}".format(group_idxs))

        ds_idxs = list()
        for group_id in group_idxs:
            for offset in range(self.group_size):
                ds_idxs.append(self.group_size * group_id + offset)

        # subsample
        #indices = indices[self.rank:self.total_size:self.num_replicas]
        #indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        #:self.total_size:self.num_replicas]
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
