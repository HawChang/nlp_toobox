#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File  :   collate_funcs.py
Author:   zhanghao(changhaw@126.com)
Date  :   21/06/25 17:26:18
Desc  :   
"""

import sys
import logging
import torch

from nlp_toolbox.utils.manual_config import InstanceName
from nlp_toolbox.utils.register import RegisterSet


def padding(indice, max_length=None, min_length=None, pad_idx=0):
    """pad 函数
    """
    if max_length is None:
        max_length = max([len(t) for t in indice])

    if min_length is not None and min_length > max_length:
        max_length = min_length

    #logging.info("max_length: {}".format(max_length))
    pad_indice = list()
    for item in indice:
    #    logging.info("item: {}".format(item))
        pad_num = max_length - len(item)
        if pad_num > 0:
            item += [pad_idx] * pad_num
        #pad = [pad_idx] * max(0, max_length - len(item))
    #    logging.info("pad: {}".format(pad))
        pad_indice.append(item)
    #pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
    return torch.tensor(pad_indice)


#@RegisterSet.collate_funcs.register
def bert_classification_collate_func(batch):
    """动态padding， batch为一部分sample
    """
    token_ids = [data[InstanceName.INPUT_IDS] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data[InstanceName.INPUT_SEGMENT_IDS] for data in batch]
    target_ids = [data[InstanceName.LABEL_IDS] for data in batch]
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)

    return {
            InstanceName.INPUT_IDS: token_ids_padded,
            InstanceName.INPUT_SEGMENT_IDS: token_type_ids_padded,
            InstanceName.LABEL_IDS: target_ids,
            }


#@RegisterSet.collate_funcs.register
def bert_infer_collate_func(batch):
    """
    动态padding， batch为一部分sample
    """
    token_ids = [data[InstanceName.INPUT_IDS] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data[InstanceName.INPUT_SEGMENT_IDS] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)

    return {
            InstanceName.INPUT_IDS: token_ids_padded,
            InstanceName.INPUT_SEGMENT_IDS: token_type_ids_padded,
            }


#@RegisterSet.collate_funcs.register
def basic_collate_func(batch):
    """
    动态padding， batch为一部分sample
    """
    collate_dict = dict()

    # 添加key
    for data_dict in batch:
        for data_name in data_dict.keys():
            collate_dict[data_name] = None
        break

    # 根据key添加数据
    for cur_data_name in collate_dict.keys():
        cur_data = [data_dict[cur_data_name] for data_dict in batch]
        assert len(cur_data) > 0, "data is empty"
        if isinstance(cur_data[0], list):
#        if cur_data_name == "second_input_ids":
#            logging.info("cur_data: {}".format(cur_data))
            cur_data = padding(cur_data)
        elif isinstance(cur_data[0], int):
            cur_data = torch.tensor(cur_data, dtype=torch.long)
        collate_dict[cur_data_name] = cur_data

    return collate_dict


if __name__ == "__main__":
    pass


