#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   basic_data_loader.py
Author:   zhanghao55@baidu.com
Date  :   21/06/24 19:39:10
Desc  :   
"""

import logging

from nlp_toolbox.utils.utils import get_dataloader
from nlp_toolbox.utils.register import RegisterSet
from nlp_toolbox.data.data_loaders.base_data_loader import BaseDataLoader
from nlp_toolbox.data.data_loaders.datasets.bert_classification import BertClassificationDataset
from nlp_toolbox.data.data_loaders.collate_funcs import bert_classification_collate_func, bert_infer_collate_func


@RegisterSet.data_loaders.register
class BertClassificationDataLoader(BaseDataLoader):
    """基础构造dataloader类
    """
    def __init__(self, inputs, config, tool_dict):
        """
        """
        super(BertClassificationDataLoader, self).__init__(inputs, config, tool_dict)
        #logging.info("vocab_size: {}".format(self.vocab_size))
        #logging.info("self attrs: {}".format(self.__dict__))

    def build(self):
        """
        """
        super(BertClassificationDataLoader, self).build()
        input_data_dict = self.make_input_data()
        logging.info("input_data_dict keys: {}".format(input_data_dict.keys()))
        for k, v in input_data_dict.items():
            logging.info("cur field: {}, type: {}, size: {}".format(
                k, type(v), len(v)))

        # 构建torch.DataSet
        cur_dataset = BertClassificationDataset(**input_data_dict)

        sampler = self.config["sampler"]
        if sampler is not None:
            sampler = RegisterSet.samplers[sampler]

        cur_collate_func = bert_infer_collate_func \
                if self.config.get("is_infer", False) else bert_classification_collate_func

        self.dataloader = get_dataloader(
                dataset=cur_dataset,
                collate_fn=cur_collate_func,
                batch_size=int(self.config["batch_size"]),
                shuffle=self.config["shuffle"],
                distributed=RegisterSet.IS_DISTRIBUTED,
                sampler=sampler,
                )


if __name__ == "__main__":
    pass


