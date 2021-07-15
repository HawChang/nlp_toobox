#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   bert_classification.py
Author:   zhanghao55@baidu.com
Date  :   21/06/25 16:04:58
Desc  :   
"""

import logging
import numpy as np

from torch.utils.data import Dataset
from nlp_toolbox.utils.manual_config import InstanceName


class BertClassificationDataset(Dataset):
    """bert预测数据集
    """
    def __init__(self, **kwargs):
        """初始化
        """
        ## 一般init函数是加载所有数据
        super(BertClassificationDataset, self).__init__()
        self.data_size = None
        self.data_list_dict = dict()
        for input_name, input_data_list in kwargs.items():
            # dataloader中的数据用numpy保存
            # 相关issue: https://github.com/pytorch/pytorch/issues/13246
            assert self.data_size is None or self.data_size == len(input_data_list), \
                    "inputs' size don't consistant, prev = {}, field {} = {}".format(
                            self.data_size, input_name, len(input_data_list))
            if self.data_size is None:
                self.data_size = len(input_data_list)
            self.data_list_dict[input_name] = np.array(input_data_list, dtype=object)
        #self.data_list = np.array(list(zip(
        #    kwargs[InstanceName.INPUT_IDS],
        #    kwargs[InstanceName.INPUT_SEGMENT_IDS],
        #    kwargs[InstanceName.LABEL_IDS],
        #    )), dtype=object)
        logging.info("data_list shape: ({},{})".format(self.data_size, len(self.data_list_dict.keys())))

    def __getitem__(self, index):
        # 得到单个数据
        output_dict = dict()
        for input_name, input_data_array in self.data_list_dict.items():
            output_dict[input_name] = input_data_array[index]
        #    in
        #token_ids, token_type_ids, label_id = self.data_list[index]
        #output = {
        #    "token_ids": token_ids,
        #    "token_type_ids": token_type_ids,
        #    "labels": label_id
        #}
        return output_dict

    def __len__(self):
        return  0 if self.data_size is None else self.data_size


if __name__ == "__main__":
    pass


