#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   adversarial_training.py
Author:   zhanghao55@baidu.com
Date  :   21/01/09 20:29:56
Desc  :   
"""

import sys
import logging
import torch


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = None

    def attack(self, epsilon=1., emb_name='emb.'):
        self.backup = dict()
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                #logging.info("cur param name: {}".format(name))
                #logging.info("param: {}".format(param))
                self.backup[name] = param.data.clone()
                #logging.info("param grad: {}".format(param.grad))
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name=None):
        """
        emb_name为None，则backup中有的param都恢复，否则只恢复包含emb_name的param
        """
        if self.backup is None:
            raise ValueError("no backup to restore, should be called after attack")

        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if emb_name is None:
                    if name in self.backup:
                        #logging.info("cur param name: {}".format(name))
                        #logging.info("param: {}".format(param))
                        param.data = self.backup[name]
                elif emb_name in name:
                    #logging.info("cur param name: {}".format(name))
                    #logging.info("param: {}".format(param))
                    assert name in self.backup
                    param.data = self.backup[name]
        self.backup = None

if __name__ == "__main__":
    pass


